# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import warnings
from contextlib import contextmanager
from typing import Callable, ContextManager

import torch
from accelerate import init_empty_weights
from megatron.core import dist_checkpointing
from megatron.core import parallel_state as mpu
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.serialization import StrictHandling
from megatron.core.models.gpt.gpt_model import ModelType
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoConfig

from verl.models.mcore import hf_to_mcore_config
from verl.utils.megatron_utils import get_model


def _init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_path", type=str, required=True, help="The path for the huggingface model")
    parser.add_argument("--output_path", type=str, required=True, help="The path for the output mcore model")
    parser.add_argument("--use_cpu_initialization", action="store_true", help="Whether to use cpu initialization")
    parser.add_argument("--test", action="store_true", help="Whether to test the conversion")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to trust remote code")
    args = parser.parse_args()
    return args


class MegatronConfig:
    def __init__(self):
        self.params_dtype = torch.bfloat16


class ModelConfig:
    def __init__(self):
        self.path = None


class Config:
    def __init__(self):
        self.model = ModelConfig()


def test_conversion(megatron_model_provider, tfconfig, output_path, model):
    ########### test ###########
    # load model
    model_test = get_model(
        model_provider_func=megatron_model_provider,
        model_type=ModelType.encoder_or_decoder,
        wrap_with_ddp=True,
        transformer_config=tfconfig,
    )
    ref_state_dict = model_test[0].module.sharded_state_dict()
    dist_checkpointing.load(ref_state_dict, output_path, strict=StrictHandling.ASSUME_OK_UNEXPECTED)

    dut_state_dict = model[0].module.state_dict()
    for name in dut_state_dict.keys():
        if dut_state_dict[name] is None:
            print(f"[Warning] {name} is none in dut_state_dict")
            continue
        dut_data = dut_state_dict[name].data
        if name in ref_state_dict:
            ref_data = ref_state_dict[name]
            if isinstance(ref_data, ShardedTensor):
                ref_data = ref_data.data.view(ref_data.local_shape)
            else:
                ref_data = ref_data.data
            assert dut_data.shape == ref_data.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            print(f"{name} is equal")
        else:
            print(f"[Warning] {name} is not in ref_state_dict")
    for name in ref_state_dict.keys():
        if ref_state_dict[name] is None:
            print(f"[Warning] {name} is none in ref_state_dict")
            continue
        ref_data = ref_state_dict[name]
        if isinstance(ref_data, ShardedTensor):
            ref_data = ref_data.data.view(ref_data.local_shape)
        else:
            ref_data = ref_data.data
        if name in dut_state_dict:
            dut_data = dut_state_dict[name].data
            assert dut_data.shape == ref_data.shape, f"{name=} {dut_data.shape=} {ref_data.shape=}"
            assert (dut_data == ref_data).all(), f"{name} is not equal"
            print(f"{name} is equal")
        else:
            print(f"[Warning] {name} is not in dut_state_dict")
    print("Conversion test passed!")


def convert_checkpoint_from_transformers_to_megatron(hf_model, model, hf_config):
    num_attention_heads = hf_config.num_attention_heads
    num_key_value_heads = hf_config.num_key_value_heads
    hidden_dim = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", hidden_dim // num_attention_heads)
    if num_attention_heads != num_key_value_heads:
        print("[WARNING] Converting GQA model")
    has_qkv_bias = getattr(hf_config, "qkv_bias", False) or getattr(hf_config, "attention_bias", False)
    has_share_expert = getattr(hf_config, "shared_expert_intermediate_size", None)
    with torch.no_grad():
        model.embedding.word_embeddings.weight.copy_(hf_model.model.embed_tokens.weight)
        for layer, hf_layer in zip(model.decoder.layers, hf_model.model.layers):
            layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.input_layernorm.weight)

            q = hf_layer.self_attn.q_proj.weight.view([num_key_value_heads, head_dim * num_attention_heads // num_key_value_heads, -1])
            k = hf_layer.self_attn.k_proj.weight.view([num_key_value_heads, head_dim, -1])
            v = hf_layer.self_attn.v_proj.weight.view([num_key_value_heads, head_dim, -1])
            qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_dim).contiguous()
            layer.self_attention.linear_qkv.weight.copy_(qkv)

            if has_qkv_bias:
                q_bias = hf_layer.self_attn.q_proj.bias.view([num_key_value_heads, -1])
                k_bias = hf_layer.self_attn.k_proj.bias.view([num_key_value_heads, -1])
                v_bias = hf_layer.self_attn.v_proj.bias.view([num_key_value_heads, -1])
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=1).view(-1).contiguous()
                layer.self_attention.linear_qkv.bias.copy_(qkv_bias)

            if hasattr(hf_layer.self_attn, "q_norm"):
                layer.self_attention.q_layernorm.weight.copy_(hf_layer.self_attn.q_norm.weight.data)
                layer.self_attention.k_layernorm.weight.copy_(hf_layer.self_attn.k_norm.weight.data)

            layer.self_attention.linear_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)
            layer.pre_mlp_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)

            layer.mlp.router.weight.copy_(hf_layer.mlp.gate.weight)

            for idx, hf_expert in enumerate(hf_layer.mlp.experts):
                fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                layer.mlp.experts.linear_fc1._parameters[f"weight{idx}"].copy_(fc1_weight)
                layer.mlp.experts.linear_fc2._parameters[f"weight{idx}"].copy_(hf_expert.down_proj.weight)

            if has_share_expert:
                layer.mlp.shared_experts.gate_weight.copy_(hf_layer.mlp.shared_expert_gate.weight)
                shared_fc1_weight = torch.cat([hf_layer.mlp.shared_expert.gate_proj.weight, hf_layer.mlp.shared_expert.up_proj.weight])
                layer.mlp.shared_experts.linear_fc1.weight.copy_(shared_fc1_weight)
                layer.mlp.shared_experts.linear_fc2.weight.copy_(hf_layer.mlp.shared_expert.down_proj.weight)

        model.decoder.final_layernorm.weight.copy_(hf_model.model.norm.weight)
        model.output_layer.weight.copy_(hf_model.lm_head.weight)


def safe_copy(
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    skip_dtype_assert: bool = False,
):
    if not skip_dtype_assert:
        if src_tensor.dtype != dst_tensor.dtype:
            raise ValueError(f"Get source dtype {src_tensor.dtype}, but target dtype {dst_tensor.dtype}")
    assert src_tensor.shape == dst_tensor.shape
    dst_tensor.data.copy_(src_tensor.data)
    return src_tensor.numel()


@torch.inference_mode()
def convert_checkpoint_from_transformers_to_megatron_qwen2_5_vl(hfmodel, mgmodel, hf_config):
    mgmodel = mgmodel.bfloat16()
    hfmodel = hfmodel.bfloat16()
    num_attention_heads = hf_config.num_attention_heads
    num_query_groups = hf_config.num_key_value_heads
    hidden_size = hf_config.hidden_size
    head_dim = hidden_size // num_attention_heads

    # 1. vision model
    hfvision = hfmodel.visual
    mgvision = mgmodel.vision_model
    vision_hidden_size = mgvision.config.hidden_size
    vision_num_query_groups = mgvision.config.num_query_groups
    vision_head_dim = vision_hidden_size // mgvision.config.num_attention_heads
    copied_numel = 0
    safe_copy(hfvision.rotary_pos_emb.inv_freq, mgvision.rotary_pos_emb.inv_freq)
    copied_numel += safe_copy(hfvision.patch_embed.proj.weight, mgvision.patch_embed.proj.weight)
    for hfblock, mgblock in zip(hfvision.blocks, mgvision.decoder.layers):
        # norm1 --> linear_qkv.norm
        copied_numel += safe_copy(hfblock.norm1.weight, mgblock.self_attention.linear_qkv.layer_norm_weight)
        # norm2 --> mlp.linear_fc1.norm
        copied_numel += safe_copy(hfblock.norm2.weight, mgblock.mlp.linear_fc1.layer_norm_weight)
        # qkv --> self_attention.linear_qkv
        converted_weight = hfblock.attn.qkv.weight.view(3, vision_num_query_groups, -1, vision_head_dim, vision_hidden_size).transpose(0, 1).flatten(1, 2).reshape(-1, vision_hidden_size).contiguous()
        copied_numel += safe_copy(converted_weight, mgblock.self_attention.linear_qkv.weight)
        converted_bias = hfblock.attn.qkv.bias.view(3, vision_num_query_groups, -1).transpose(0, 1).flatten(1, 2).view(-1).contiguous()
        copied_numel += safe_copy(converted_bias, mgblock.self_attention.linear_qkv.bias)
        # proj --> self_attention.linear_proj
        copied_numel += safe_copy(hfblock.attn.proj.weight, mgblock.self_attention.linear_proj.weight)
        copied_numel += safe_copy(hfblock.attn.proj.bias, mgblock.self_attention.linear_proj.bias)
        # mlp --> mlp: gate
        fc1_weight = torch.cat([hfblock.mlp.gate_proj.weight, hfblock.mlp.up_proj.weight])
        fc1_bias = torch.cat([hfblock.mlp.gate_proj.bias, hfblock.mlp.up_proj.bias])
        copied_numel += safe_copy(fc1_weight, mgblock.mlp.linear_fc1.weight)
        copied_numel += safe_copy(fc1_bias, mgblock.mlp.linear_fc1.bias)
        copied_numel += safe_copy(hfblock.mlp.down_proj.weight, mgblock.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hfblock.mlp.down_proj.bias, mgblock.mlp.linear_fc2.bias)

    # 2. vision projector
    hfprojector = hfvision.merger
    mgprojector = mgvision.projection
    copied_numel += safe_copy(hfprojector.ln_q.weight, mgvision.decoder.final_layernorm.weight)

    copied_numel += safe_copy(hfprojector.mlp[0].weight, mgprojector.encoder.linear_fc1.weight)
    copied_numel += safe_copy(hfprojector.mlp[0].bias, mgprojector.encoder.linear_fc1.bias)
    copied_numel += safe_copy(hfprojector.mlp[2].weight, mgprojector.encoder.linear_fc2.weight)
    copied_numel += safe_copy(hfprojector.mlp[2].bias, mgprojector.encoder.linear_fc2.bias)
    n_params = sum([t.numel() for t in hfvision.state_dict().values()])
    assert n_params == copied_numel
    # 3. llm [just Qwen2]
    hfllm = hfmodel.model
    mgllm = mgmodel.language_model
    copied_numel = 0
    copied_numel += safe_copy(hfllm.embed_tokens.weight, mgllm.embedding.word_embeddings.weight)
    for mglayer, hflayer in zip(mgllm.decoder.layers, hfllm.layers):
        copied_numel += safe_copy(hflayer.input_layernorm.weight, mglayer.self_attention.linear_qkv.layer_norm_weight)

        q_proj_weight = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        k_proj_weight = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        v_proj_weight = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        qkv_proj = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=1).view(-1, hidden_size).contiguous()
        copied_numel += safe_copy(qkv_proj, mglayer.self_attention.linear_qkv.weight)

        q_proj_bias = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
        k_proj_bias = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
        v_proj_bias = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
        qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=1).view(-1).contiguous()
        copied_numel += safe_copy(qkv_bias, mglayer.self_attention.linear_qkv.bias)
        copied_numel += safe_copy(hflayer.self_attn.o_proj.weight, mglayer.self_attention.linear_proj.weight)

        fc1_weight = torch.cat([hflayer.mlp.gate_proj.weight, hflayer.mlp.up_proj.weight])
        copied_numel += safe_copy(fc1_weight, mglayer.mlp.linear_fc1.weight)

        copied_numel += safe_copy(hflayer.mlp.down_proj.weight, mglayer.mlp.linear_fc2.weight)
        copied_numel += safe_copy(hflayer.post_attention_layernorm.weight, mglayer.mlp.linear_fc1.layer_norm_weight)

    copied_numel += safe_copy(hfllm.norm.weight, mgllm.decoder.final_layernorm.weight)
    if not hf_config.tie_word_embeddings:
        safe_copy(hfmodel.lm_head.weight, mgllm.output_layer.weight)

    n_params = sum([t.numel() for t in hfllm.state_dict().values()])

    assert n_params == copied_numel


@torch.no_grad()
def convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model, hf_config, tfconfig):
    warnings.warn("MTP model is not supported yet", stacklevel=2)
    numel: int = 0
    numel += safe_copy(hf_model.model.embed_tokens.weight, model.embedding.word_embeddings.weight)
    print(f"{numel=}")
    for layer_idx, (layer, hf_layer) in enumerate(zip(model.decoder.layers, hf_model.model.layers)):
        numel_cur: int = numel
        numel += safe_copy(hf_layer.input_layernorm.weight, layer.input_layernorm.weight)

        if hf_config.q_lora_rank is None:
            numel += safe_copy(hf_layer.self_attn.q_proj.weight, layer.self_attention.linear_q_proj.weight)
        else:
            numel += safe_copy(hf_layer.self_attn.q_a_proj.weight, layer.self_attention.linear_q_down_proj.weight)
            numel += safe_copy(hf_layer.self_attn.q_b_proj.weight, layer.self_attention.linear_q_up_proj.weight)
            numel += safe_copy(hf_layer.self_attn.q_a_layernorm.weight, layer.self_attention.linear_q_up_proj.layer_norm_weight)

        numel += safe_copy(hf_layer.self_attn.kv_a_proj_with_mqa.weight, layer.self_attention.linear_kv_down_proj.weight)
        numel += safe_copy(hf_layer.self_attn.kv_b_proj.weight, layer.self_attention.linear_kv_up_proj.weight)
        numel += safe_copy(hf_layer.self_attn.kv_a_layernorm.weight, layer.self_attention.linear_kv_up_proj.layer_norm_weight)
        numel += safe_copy(hf_layer.self_attn.o_proj.weight, layer.self_attention.linear_proj.weight)

        if not hasattr(layer.mlp, "router"):
            numel += safe_copy(hf_layer.post_attention_layernorm.weight, layer.mlp.linear_fc1.layer_norm_weight)
            numel += safe_copy(torch.cat([hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight]), layer.mlp.linear_fc1.weight)
            numel += safe_copy(hf_layer.mlp.down_proj.weight, layer.mlp.linear_fc2.weight)
        else:
            numel += safe_copy(hf_layer.mlp.gate.weight, layer.mlp.router.weight)
            # NOTE: the e_score_correction_bias in mcore model will be initialized with bfloat16 and \
            # recover to fp32 in the first forward. There is always a diff in the bias between two models (~0.3%)
            numel += safe_copy(hf_layer.mlp.gate.e_score_correction_bias, layer.mlp.router.expert_bias, skip_dtype_assert=True)
            if tfconfig.moe_grouped_gemm:
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    linear_fc1_weighti = getattr(layer.mlp.experts.linear_fc1, "weight" + str(i))
                    numel += safe_copy(fc1_weight, linear_fc1_weighti)
                    linear_fc2_weighti = getattr(layer.mlp.experts.linear_fc2, "weight" + str(i))
                    numel += safe_copy(hf_expert.down_proj.weight, linear_fc2_weighti)
            else:
                for i, hf_expert in enumerate(hf_layer.mlp.experts):
                    expert = layer.mlp.experts.local_experts[i]
                    fc1_weight = torch.cat([hf_expert.gate_proj.weight, hf_expert.up_proj.weight])
                    numel += safe_copy(fc1_weight, expert.linear_fc1.weight)
                    numel += safe_copy(hf_expert.down_proj.weight, expert.linear_fc2.weight)
            numel += safe_copy(hf_layer.post_attention_layernorm.weight, layer.pre_mlp_layernorm.weight)
            shared_fc1_weight = torch.cat([hf_layer.mlp.shared_experts.gate_proj.weight, hf_layer.mlp.shared_experts.up_proj.weight])
            numel += safe_copy(shared_fc1_weight, layer.mlp.shared_experts.linear_fc1.weight)
            numel += safe_copy(hf_layer.mlp.shared_experts.down_proj.weight, layer.mlp.shared_experts.linear_fc2.weight)
            print(f"{layer_idx=} {numel=} numel this layer={numel - numel_cur}")

    numel += safe_copy(hf_model.model.norm.weight, model.decoder.final_layernorm.weight)

    if not hf_config.tie_word_embeddings:
        numel += safe_copy(hf_model.lm_head.weight, model.output_layer.weight)
    print(f"{numel=}")
    return numel


@contextmanager
def noop_context() -> None:
    yield


def convert_hf_to_mcore(hf_model_path, output_path, use_cpu_initialization=False, test=False, trust_remote_code=False):
    os.makedirs(output_path, exist_ok=True)
    if len(os.listdir(output_path)) > 0 and not test:
        print(f"Output path {output_path} is not empty, skipping conversion")
        return

    # init torch distributed and mpu
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

    # init hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path)
    print(hf_config, flush=True)

    cfg = Config()
    cfg.model.path = hf_model_path
    tfconfig = hf_to_mcore_config(hf_config, torch.bfloat16)
    tfconfig.use_cpu_initialization = use_cpu_initialization
    tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)

    # init megatron model
    def megatron_model_provider(pre_process, post_process):
        from verl.models.mcore import init_mcore_model

        parallel_model = init_mcore_model(
            tfconfig,
            hf_config,
            pre_process,
            post_process,
            share_embeddings_and_output_weights=tie_word_embeddings,
            value=False,
        )
        return parallel_model

    context: Callable[..., ContextManager] = init_empty_weights if use_cpu_initialization else noop_context
    with context():
        model = get_model(
            model_provider_func=megatron_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            transformer_config=tfconfig,
        )

    if use_cpu_initialization:
        # convert meta device to empty tensor so it can use `copy_` function
        model[0].module = model[0].module.to_empty(device="cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # init hf model
    if "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
        hf_model = AutoModelForImageTextToText.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
    hf_state_dict = hf_model.state_dict()

    # load hf state dict to megatron model
    if "Qwen2MoeForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    elif "Qwen2_5_VLForConditionalGeneration" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron_qwen2_5_vl(hf_model, model[0].module, hf_config)
    elif "DeepseekV3ForCausalLM" in hf_config.architectures:
        numel: int = convert_checkpoint_from_transformers_to_megatron_dpskv3(hf_model, model[0].module, hf_config, tfconfig=tfconfig)
        if numel != hf_model.num_parameters():
            warnings.warn(f"numel mismatch: {numel=} != {hf_model.num_parameters()=}", stacklevel=1)
    elif "Qwen3MoeForCausalLM" in hf_config.architectures:
        convert_checkpoint_from_transformers_to_megatron(hf_model, model[0].module, hf_config)
    else:
        assert not use_cpu_initialization, "use_cpu_initialization is only supported for MoE model"
        from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel

        load_state_dict_to_megatron_gptmodel(
            state_dict=hf_state_dict,
            wrapped_models=model,
            config=hf_config,
            params_dtype=torch.bfloat16,
            is_value_model=False,
        )

    megatron_state_dict = model[0].module.sharded_state_dict()
    del hf_state_dict, hf_model

    # save megatron model
    if len(os.listdir(output_path)) == 0:
        dist_checkpointing.save(megatron_state_dict, output_path, sharded_strategy=None, async_sharded_save=False)
    if test:
        test_conversion(megatron_model_provider, tfconfig, output_path, model)


if __name__ == "__main__":
    args = _init_args()
    convert_hf_to_mcore(args.hf_model_path, args.output_path, args.use_cpu_initialization, args.test, args.trust_remote_code)
