# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True, return_hist_data: bool = False):
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - response_length_only_model/mean, max, min: Statistics about model-only response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    model_only_response_mask = response_mask
    # 计算仅模型输出的长度（基于loss_mask）
    if "loss_mask" in batch.batch:
        # loss_mask中值为1的部分是模型的实际输出，需要计算损失的部分
        response_loss_mask = batch.batch["loss_mask"][:, -max_response_length:].bool()
        # 只计算loss_mask为1且attention_mask也为1的token数量
        model_only_response_mask = response_mask & response_loss_mask
        response_length_only_model = model_only_response_mask.sum(-1).float()
    else:
        # 如果没有loss_mask，fallback到原来的response_length
        response_length_only_model = response_length

    # 多样性计算
    def compute_sample_diversity_metrics(batch, model_only_response_mask):
        """计算采样多样性指标"""
        responses = batch.batch["responses"]
        bsz = len(batch.batch)
        index = batch.non_tensor_batch["uid"]
        
        # 提取模型输出并按query分组
        id2response = defaultdict(list)
        id2first_turn_response = defaultdict(list)
        first_turn_response_mask = extract_first_turn_mask_optimized(model_only_response_mask)
        for i in range(bsz):
            if model_only_response_mask[i].sum() > 0:  # 只处理非空响应
                valid_tokens = responses[i][model_only_response_mask[i]]
                id2response[index[i]].append(valid_tokens)
            # 提取第一轮回答
            if first_turn_response_mask[i].sum() > 0:
                first_turn_tokens = responses[i][first_turn_response_mask[i]]
                id2first_turn_response[index[i]].append(first_turn_tokens)

        # 计算N-gram多样性
        def compute_ngram_diversity(response_list, N=4):
            """N-gram多样性计算"""
            if not response_list:
                return 0.0
                
            all_ngrams = []
            for response_tokens in response_list:
                if len(response_tokens) >= N:
                    ngrams = [tuple(response_tokens[i:i + N].tolist())
                             for i in range(len(response_tokens) - N + 1)]
                    all_ngrams.extend(ngrams)
            
            if not all_ngrams:
                return 0.0
                
            unique_ngrams = len(set(all_ngrams))
            return unique_ngrams / len(all_ngrams)
        
        # 计算序列多样性
        def compute_sequence_diversity(response_list):
            """优化的序列多样性计算"""
            if not response_list:
                return 0, 0.0
                
            # 使用set推导式提高效率
            unique_sequences = {tuple(tokens.tolist()) for tokens in response_list}
            unique_count = len(unique_sequences)
            unique_ratio = unique_count / len(response_list)
            
            return unique_count, unique_ratio
        
        # 计算指标
        N = 4
        query_ngram_ratios = []
        query_sequence_counts = []
        query_sequence_ratios = []
        
        for query_id, response_list in id2response.items():
            # 只计算有多个响应的query的多样性
            if len(response_list) > 1:
                # N-gram多样性
                ngram_ratio = compute_ngram_diversity(response_list, N)
                query_ngram_ratios.append(ngram_ratio)
                # 序列多样性
                seq_count, seq_ratio = compute_sequence_diversity(response_list)
                query_sequence_counts.append(seq_count)
                query_sequence_ratios.append(seq_ratio)

        # 计算第一轮回答的多样性
        first_turn_ngram_ratios = []
        first_turn_sequence_counts = []
        first_turn_sequence_ratios = []
        for query_id, first_turn_response_list in id2first_turn_response.items():
            # 计算第一轮回答的序列多样性
            if len(first_turn_response_list) > 1:
                # N-gram多样性
                ngram_ratio = compute_ngram_diversity(first_turn_response_list, N)
                first_turn_ngram_ratios.append(ngram_ratio)
                # 序列多样性
                seq_count, seq_ratio = compute_sequence_diversity(first_turn_response_list)
                first_turn_sequence_counts.append(seq_count)
                first_turn_sequence_ratios.append(seq_ratio)
        
        # 计算统计值
        def safe_stats(values):
            if not values:
                return 0.0, 0.0, 0.0
            return np.mean(values), np.max(values), np.min(values)

        def safe_array(values):
            return np.array(values) if values else np.array([])
        
        ngram_mean, ngram_max, ngram_min = safe_stats(query_ngram_ratios)
        seq_ratio_mean, seq_ratio_max, seq_ratio_min = safe_stats(query_sequence_ratios)
        seq_count_mean, seq_count_max, seq_count_min = safe_stats(query_sequence_counts)
        # 计算第一轮统计值
        first_turn_ngram_mean, first_turn_ngram_max, first_turn_ngram_min = safe_stats(first_turn_ngram_ratios)
        first_turn_seq_ratio_mean, first_turn_seq_ratio_max, first_turn_seq_ratio_min = safe_stats(
            first_turn_sequence_ratios)
        first_turn_seq_count_mean, first_turn_seq_count_max, first_turn_seq_count_min = safe_stats(
            first_turn_sequence_counts)
        
        scalar_metrics = {
            f"sample_diversity/unique_{N}gram_ratio_mean": ngram_mean,
            f"sample_diversity/unique_{N}gram_ratio_max": ngram_max,
            f"sample_diversity/unique_{N}gram_ratio_min": ngram_min,
            f"sample_diversity/unique_sequence_ratio_mean": seq_ratio_mean,
            f"sample_diversity/unique_sequence_ratio_max": seq_ratio_max,
            f"sample_diversity/unique_sequence_ratio_min": seq_ratio_min,
            f"sample_diversity/unique_sequence_count_mean": seq_count_mean,
            f"sample_diversity/unique_sequence_count_max": seq_count_max,
            f"sample_diversity/unique_sequence_count_min": seq_count_min,
            # 添加有效query数量统计
            f"sample_diversity/valid_query_count": len(query_ngram_ratios),
            # 添加第一轮多样性指标
            f"sample_diversity/first_turn_unique_{N}gram_ratio_mean": first_turn_ngram_mean,
            f"sample_diversity/first_turn_unique_{N}gram_ratio_max": first_turn_ngram_max,
            f"sample_diversity/first_turn_unique_{N}gram_ratio_min": first_turn_ngram_min,
            f"sample_diversity/first_turn_unique_sequence_ratio_mean": first_turn_seq_ratio_mean,
            f"sample_diversity/first_turn_unique_sequence_ratio_max": first_turn_seq_ratio_max,
            f"sample_diversity/first_turn_unique_sequence_ratio_min": first_turn_seq_ratio_min,
            f"sample_diversity/first_turn_unique_sequence_count_mean": first_turn_seq_count_mean,
            f"sample_diversity/first_turn_unique_sequence_count_max": first_turn_seq_count_max,
            f"sample_diversity/first_turn_unique_sequence_count_min": first_turn_seq_count_min,
        }

        hist_metrics = {
            "sample_diversity/unique_ngram_ratio_hist": safe_array(query_ngram_ratios),
            "sample_diversity/unique_sequence_ratio_hist": safe_array(query_sequence_ratios),
            "sample_diversity/unique_sequence_count_hist": safe_array(query_sequence_counts),
            "sample_diversity/first_turn_unique_ngram_ratio_hist": safe_array(first_turn_ngram_ratios),
            "sample_diversity/first_turn_unique_sequence_ratio_hist": safe_array(first_turn_sequence_ratios),
            "sample_diversity/first_turn_unique_sequence_count_hist": safe_array(first_turn_sequence_counts),
        }
        return scalar_metrics, hist_metrics

    diversity_scalar_metrics, diversity_hist_metrics = compute_sample_diversity_metrics(batch, model_only_response_mask)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    tool_call_times = batch.batch["tool_call_times"]
    rollout_turns = batch.batch["rollout_turns"]  # 添加 turns 变量
    # 计算每个工具的调用次数
    tool_names = [key.removeprefix("tool_call_times_") for key in batch.batch.keys() if key.startswith("tool_call_times_")]
    tool_call_times_by_tool = {tool_name: batch.batch[f"tool_call_times_{tool_name}"] for tool_name in tool_names}

    for tool_name, tool_call_times_per_tool in tool_call_times_by_tool.items():
        if not torch.is_tensor(tool_call_times_per_tool):
            tool_call_times_by_tool[tool_name] = torch.tensor(tool_call_times_per_tool)

    if not torch.is_tensor(tool_call_times):
        tool_call_times = torch.tensor(tool_call_times)

    if not torch.is_tensor(rollout_turns):
        rollout_turns = torch.tensor(rollout_turns)

    # 计算使用工具的样本比例
    tool_usage_ratio = torch.mean((tool_call_times > 0).float()).detach().item()

    # 分工具统计，计算每个工具的调用比例
    tool_usage_ratio_by_tool = {}
    for tool_name, tool_call_times_per_tool in tool_call_times_by_tool.items():
        tool_usage_ratio_by_tool[tool_name] = torch.mean((tool_call_times_per_tool > 0).float()).detach().item()

    # 计算有使用工具的样例中，平均每个样例调用工具的次数
    tool_using_samples = tool_call_times > 0
    if torch.sum(tool_using_samples) > 0:
        tool_call_times_when_used = torch.mean(tool_call_times[tool_using_samples].float()).detach().item()
    else:
        tool_call_times_when_used = 0.0

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # 分组统计每个query的rollout_turns和tool_call_times
    # 每个query的rollout_turns统计
    index = batch.non_tensor_batch["uid"]
    id2rollout_turns = defaultdict(list)
    id2tool_call_times = defaultdict(list)
    bsz = len(batch.batch)
    for i in range(bsz):
        id2rollout_turns[index[i]].append(rollout_turns[i])
        id2tool_call_times[index[i]].append(tool_call_times[i])

    # 计算每个query的rollout_turns的均值和最大值
    id2mean_rollout_turns = {k: torch.mean(torch.tensor(v).float()).item() for k, v in id2rollout_turns.items()}
    id2max_rollout_turns = {k: torch.max(torch.tensor(v)).item() for k, v in id2rollout_turns.items()}

    # 计算每个query的tool_call_times的均值和最大值
    id2mean_tool_call_times = {k: torch.mean(torch.tensor(v).float()).item() for k, v in id2tool_call_times.items()}
    id2max_tool_call_times = {k: torch.max(torch.tensor(v)).item() for k, v in id2tool_call_times.items()}

    # 计算query level的统计信息
    # query_level_rollout_turns_mean = torch.mean(torch.tensor(list(id2mean_rollout_turns.values())).float()).detach().item()
    # query_level_tool_call_times_mean = torch.mean(torch.tensor(list(id2mean_tool_call_times.values())).float()).detach().item()

    # 分组统计每个query的正确/错误轨迹数量
    scores = batch.batch["token_level_rewards"].sum(dim=-1)
    id2score = defaultdict(list)
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
    # 计算query level的score的均值
    id2mean_score = {k: torch.mean(torch.tensor(v).float()).item() for k, v in id2score.items()}
    # query_level_score_mean = torch.mean(torch.tensor(list(id2mean_score.values())).float()).detach().item()
    # 计算query level的正确/错误轨迹数量，比例
    correct_score, wrong_score = 1, 0 # 暂定正确和错误轨迹的得分分别为1，0
    id2correct_trajectories_num = {k: torch.sum(torch.tensor(v) >= correct_score).item() for k, v in id2score.items()}
    id2correct_trajectories_ratio = {k: v / len(id2score[k]) for k, v in id2correct_trajectories_num.items()}
    id2middle_trajectories_num = {k: torch.sum((torch.tensor(v) > wrong_score) & (torch.tensor(v) < correct_score)).item() for k, v in id2score.items()}
    id2middle_trajectories_ratio = {k: v / len(id2score[k]) for k, v in id2middle_trajectories_num.items()}
    id2wrong_trajectories_num = {k: torch.sum(torch.tensor(v) <= wrong_score).item() for k, v in id2score.items()}
    id2wrong_trajectories_ratio = {k: v / len(id2score[k]) for k, v in id2wrong_trajectories_num.items()}
    # 采样全部正确/错误的query数量和比例
    all_correct_query_num = sum(1 for v in id2score.values() if all(score >= correct_score for score in v))
    all_correct_query_ratio = all_correct_query_num / len(id2score)
    all_wrong_query_num = sum(1 for v in id2score.values() if all(score <= wrong_score for score in v))
    all_wrong_query_ratio = all_wrong_query_num / len(id2score)
    valid_query_num =  len(id2score) - all_wrong_query_num - all_correct_query_num
    valid_query_ratio = valid_query_num / len(id2score)

    # 构建工具指标
    tool_metrics = {}
    for tool_name, tool_times in tool_call_times_by_tool.items():
        prefix = f"tool_call/{tool_name}"
        tool_metrics.update({
            f"{prefix}/times/mean": torch.mean(tool_times.float()).detach().item(),
            f"{prefix}/times/max": torch.max(tool_times.float()).detach().item(),
            f"{prefix}/times/min": torch.min(tool_times.float()).detach().item(),
            f"{prefix}/usage_ratio": tool_usage_ratio_by_tool[tool_name]
        })

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length (包含工具返回结果)
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # response length only model (仅模型输出，不包含工具返回结果)
        "response_length_only_model/mean": torch.mean(response_length_only_model).detach().item(),
        "response_length_only_model/max": torch.max(response_length_only_model).detach().item(),
        "response_length_only_model/min": torch.min(response_length_only_model).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # tool using
        "tool_call/times/mean": torch.mean(tool_call_times.float()).detach().item(),
        "tool_call/times/max": torch.max(tool_call_times).detach().item(),
        "tool_call/times/min": torch.min(tool_call_times).detach().item(),
        "tool_call/usage_ratio": tool_usage_ratio,  # 使用工具的样本比例
        "tool_call/times_when_used": tool_call_times_when_used,  # 有使用工具时，每次的平均调用次数
        # rollout turns
        "rollout/turns/mean": torch.mean(rollout_turns.float()).detach().item(),
        "rollout/turns/max": torch.max(rollout_turns).detach().item(),
        "rollout/turns/min": torch.min(rollout_turns).detach().item(),
        # query level统计
        # "rollout/turns/query_level_mean": query_level_rollout_turns_mean, # 每query的采样量一致是和全局统计值一致，暂不展示
        # "tool_call/times/query_level_mean": query_level_tool_call_times_mean, # 每query的采样量一致是和全局统计值一致，暂不展示
        # "critic/score/query_level_mean": query_level_score_mean, # 每query的采样量一致是和全局统计值一致，暂不展示
        "trajectories/valid_query_num": valid_query_num,
        "trajectories/valid_query_ratio": valid_query_ratio,
        "trajectories/all_correct_query_num": all_correct_query_num,
        "trajectories/all_correct_query_ratio": all_correct_query_ratio,
        "trajectories/all_wrong_query_num": all_wrong_query_num,
        "trajectories/all_wrong_query_ratio": all_wrong_query_ratio,
        **tool_metrics,
    }
    metrics.update(diversity_scalar_metrics)
    # 如果不需要直方图数据，直接返回原始指标
    if not return_hist_data:
        return metrics
    # 为TensorBoard准备直方图数据
    hist_data = {
        "critic/score_hist": sequence_score.detach().cpu().numpy(),
        "critic/rewards_hist": sequence_reward.detach().cpu().numpy(),
        "critic/advantages_hist": valid_adv.detach().cpu().numpy(),
        "critic/returns_hist": valid_returns.detach().cpu().numpy(),
        "response_length_hist": response_length.detach().cpu().numpy(),
        "response_length_only_model_hist": response_length_only_model.detach().cpu().numpy(),
        "prompt_length_hist": prompt_length.detach().cpu().numpy(),
        "tool_call/times_hist": tool_call_times.float().detach().cpu().numpy(),
        "rollout/turns_hist": rollout_turns.float().detach().cpu().numpy(),
        # query level直方图数据
        "rollout/turns/query_level_mean_hist": np.array(list(id2mean_rollout_turns.values()), dtype=np.float32),
        "rollout/turns/query_level_max_hist": np.array(list(id2max_rollout_turns.values())),
        "tool_call/times/query_level_mean_hist": np.array(list(id2mean_tool_call_times.values()), dtype=np.float32),
        "tool_call/times/query_level_max_hist": np.array(list(id2max_tool_call_times.values())),
        "critic/score/query_level_mean_hist": np.array(list(id2mean_score.values()), dtype=np.float32),
        "trajectories/query_level_correct_num_hist": np.array(list(id2correct_trajectories_num.values())),
        "trajectories/query_level_correct_ratio_hist": np.array(list(id2correct_trajectories_ratio.values())),
        "trajectories/query_level_wrong_num_hist": np.array(list(id2wrong_trajectories_num.values())),
        "trajectories/query_level_middle_ratio_hist": np.array(list(id2middle_trajectories_ratio.values())),
        "trajectories/query_level_middle_num_hist": np.array(list(id2middle_trajectories_num.values())),
        "trajectories/query_level_wrong_ratio_hist": np.array(list(id2wrong_trajectories_ratio.values())),
        **(
            {
                f"tool_call/{tool_name}/times_hist": tool_call_times_per_tool.float().detach().cpu().numpy()
                for tool_name, tool_call_times_per_tool in tool_call_times_by_tool.items()
            }
        )
    }
    hist_data.update(diversity_hist_metrics)
    return metrics, hist_data


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val


def extract_first_turn_mask_optimized(response_mask):
    """
    提取第一轮回答的mask，即从左到右第一组连续的True
    假设：对于非空的response mask，第一个元素必须是True

    Args:
        response_mask: 布尔tensor，形状为(batch_size, seq_len)

    Returns:
        first_turn_mask: 只保留第一组连续True的mask
    """
    # 复制原始mask
    first_turn_mask = response_mask.clone()

    # 检查第一列是否都为True（可选的断言检查）
    assert torch.all(response_mask[:, 0][response_mask.sum(dim=1) > 0]), "First element of non-empty masks must be True"

    # 找出每个样本中第一个False的位置
    # 先将response_mask取反，得到False的位置为True
    inverted_mask = ~response_mask

    # 对于每个样本，找到第一个True(即原始mask中的False)的位置
    # 使用cumsum来标记第一个True之后的所有位置
    # 对于全True的行(原始mask中全False)，cumsum会将所有位置都标记为>0
    first_false_marker = torch.cumsum(inverted_mask, dim=1)

    # 将第一个False之后的所有位置设为False
    # 只要first_false_marker > 0，说明已经遇到了至少一个False
    first_turn_mask[first_false_marker > 0] = False

    return first_turn_mask