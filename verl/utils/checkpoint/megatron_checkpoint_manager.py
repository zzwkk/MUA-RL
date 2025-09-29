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

import logging
import os
import random

import numpy as np
import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from omegaconf import DictConfig
from transformers import GenerationConfig

from verl.models.weight_loader_registry import get_weight_saver
from verl.utils.device import is_cuda_available, is_npu_available
from verl.utils.fs import is_non_local
from verl.utils.logger import log_with_rank
from verl.utils.megatron_utils import (
    get_hf_config_and_tokenizer_checkpoint_path,
    get_hf_model_checkpoint_path,
    get_model_checkpoint_path,
    get_optimizer_checkpoint_path,
    get_optimizer_scheduler_checkpoint_path,
    get_rng_states_checkpoint_path,
)

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class MegatronCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer/processor and config for ckpt merge
    """

    def __init__(
        self,
        config,
        model_config,
        role,
        model: torch.nn.ModuleList,
        arch: str,
        hf_config,
        param_dtype: torch.dtype,
        share_embeddings_and_output_weights: bool,
        tokenizer,
        optimizer,
        optimizer_scheduler,
        use_distributed_optimizer: bool,
        use_checkpoint_opt_param_scheduler: bool = False,
        checkpoint_contents: DictConfig = None,
        **kwargs,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            lr_scheduler=optimizer_scheduler,
            processing_class=tokenizer,
            checkpoint_contents=checkpoint_contents,
        )
        self.arch = arch
        self.config = config
        self.role = role
        self.is_value_model = False
        if self.role in ["reward", "critic"]:
            self.is_value_model = True
        self.model_config = model_config
        self.hf_config = hf_config
        self.param_dtype = param_dtype
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.model_path = self.config.model.path
        self.use_distributed_optimizer = use_distributed_optimizer
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler

        self.rank = torch.distributed.get_rank()

        self.weight_saver = get_weight_saver(self.arch)

    def get_rng_state(self, use_dist_ckpt: bool = False, data_parallel_random_init: bool = False):
        """collect rng state across data parallel ranks"""
        rng_state = {
            "random_rng_state": random.getstate(),
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
        }

        if is_cuda_available:
            rng_state["cuda_rng_state"] = torch.cuda.get_rng_state()
        elif is_npu_available:
            rng_state["npu_rng_state"] = torch.npu.get_rng_state()

        rng_state_list = None
        if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
            rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
        else:
            rng_state_list = [rng_state]

        if use_dist_ckpt:
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            pp_size = mpu.get_pipeline_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            cp_rank = mpu.get_context_parallel_rank()
            cp_size = mpu.get_context_parallel_world_size()
            rng_state_list = ShardedObject(
                "rng_state",
                rng_state_list,
                (pp_size, tp_size, cp_size),
                (pp_rank, tp_rank, cp_rank),
                replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
            )

        return rng_state_list

    def get_checkpoint_name(
        self,
        checkpoints_path,
        pipeline_parallel=None,
        tensor_rank=None,
        pipeline_rank=None,
        cp_rank=None,
        expert_parallel=None,
        expert_rank=None,
        return_base_dir=True,
        basename="model.pt",
    ):
        """Determine the directory name for this rank's checkpoint."""
        # Use both the tensor and pipeline MP rank.
        if pipeline_parallel is None:
            pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        if tensor_rank is None:
            tensor_rank = mpu.get_tensor_model_parallel_rank()
        if pipeline_rank is None:
            pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        if cp_rank is None:
            cp_rank = mpu.get_context_parallel_rank()
        if expert_parallel is None:
            expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
        if expert_rank is None:
            expert_rank = mpu.get_expert_model_parallel_rank()

        # Use both the tensor and pipeline MP rank. If using the distributed
        # optimizer, then the optimizer's path must additionally include the
        # data parallel rank.

        # due to the fact that models are identical across cp ranks, cp rank is not used in the checkpoint path
        if not pipeline_parallel:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}")
        else:
            common_path = os.path.join(checkpoints_path, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

        if expert_parallel:
            common_path = common_path + f"_{expert_rank:03d}"

        os.makedirs(common_path, exist_ok=True)

        if return_base_dir:
            return common_path
        return os.path.join(common_path, basename)

    def load_optimizer(self, ckpt_path):
        # TODO: Check Optimizer format and distributed optimizer
        optimizer_path = get_optimizer_checkpoint_path(ckpt_path)
        log_with_rank(f"Loading optimizer from {optimizer_path}", rank=self.rank, logger=logger)
        self.optimizer.load_parameter_state(optimizer_path)

    def load_rng_states(self, ckpt_path, data_parallel_random_init=False, use_dist_ckpt=False):
        rng_state_path = get_rng_states_checkpoint_path(ckpt_path, only_rank0_save=False)
        log_with_rank(f"Loading rng states from {rng_state_path}", rank=self.rank, logger=logger)
        rng_state = torch.load(rng_state_path, weights_only=False)
        # access rng_state for data parallel rank
        if not use_dist_ckpt:
            rng_state = rng_state[mpu.get_data_parallel_rank()] if data_parallel_random_init else rng_state[0]
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])

        if is_cuda_available:
            torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
        elif is_npu_available:
            torch.npu.set_rng_state(rng_state["npu_rng_state"])

        # Check for empty states array
        if not rng_state["rng_tracker_states"]:
            raise KeyError
        tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        if local_path is None:
            return

        if self.should_load_model:
            model_path = get_model_checkpoint_path(local_path)
            ckpt_name = self.get_checkpoint_name(model_path, return_base_dir=False)
            state_dicts = torch.load(os.path.join(ckpt_name), weights_only=False)
            assert len(state_dicts) == len(self.model), f"state_dicts length: {len(state_dicts)} mismatch with model length: {len(self.model)}"
            for vpp_rank, (state_dict, model) in enumerate(zip(state_dicts, self.model)):
                model.load_state_dict(state_dict)
            log_with_rank(f"Loaded sharded model checkpoint from {model_path}", rank=self.rank, logger=logger)

        if self.should_load_optimizer:
            self.load_optimizer(local_path)

        if self.should_load_extra:
            self.load_rng_states(local_path)
            if self.use_checkpoint_opt_param_scheduler:
                optimizer_scheduler_path = get_optimizer_scheduler_checkpoint_path(local_path, only_rank0_save=True)
                if os.path.exists(optimizer_scheduler_path):
                    log_with_rank(f"Loading optimizer scheduler from {optimizer_scheduler_path}", rank=self.rank, logger=logger)
                    state_dict = torch.load(optimizer_scheduler_path, weights_only=False)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.load_state_dict(state_dict)
                else:
                    log_with_rank(f"Optimizer scheduler path {optimizer_scheduler_path} does not exist, skipping loading.", rank=self.rank, logger=logger)

        if del_local_after_load:
            try:
                os.remove(local_path) if is_non_local(local_path) else None
            except Exception as e:
                log_with_rank(f"remove local resume ckpt file after loading failed, exception {e} will be ignored", rank=self.rank, logger=logger)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)

        # Save Model
        if self.should_save_model and mpu.get_data_parallel_rank() == 0:
            state_dicts = []

            for vpp_rank, model in enumerate(self.model):
                state_dict = model.state_dict()
                state_dicts.append(state_dict)

            log_with_rank(f"Saving sharded model checkpoint to {local_path}", rank=self.rank, logger=logger)
            model_ckpt_path = get_model_checkpoint_path(local_path)
            hf_config_and_tokenizer_path = get_hf_config_and_tokenizer_checkpoint_path(local_path)
            ckpt_name = self.get_checkpoint_name(model_ckpt_path, return_base_dir=False)
            torch.save(state_dicts, os.path.join(ckpt_name))

            log_with_rank(f"Saved checkpoint to {model_ckpt_path}", rank=self.rank, logger=logger)
            if self.rank == 0:
                self.processing_class.save_pretrained(hf_config_and_tokenizer_path)
                self.hf_config.save_pretrained(hf_config_and_tokenizer_path)
                if hasattr(self.hf_config, "name_or_path") and self.hf_config.name_or_path:
                    try:
                        generation_config = GenerationConfig.from_pretrained(self.hf_config.name_or_path)
                        generation_config.save_pretrained(hf_config_and_tokenizer_path)
                    except Exception:
                        # if the generation config isn't available, we don't save it
                        pass
                if hdfs_path is not None:
                    log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger)
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                    hdfs_io.copy(src=hf_config_and_tokenizer_path, dst=hdfs_path, dirs_exist_ok=True)

        if self.should_save_hf_model:
            # wait for everyone to dump to local
            state_dict = self.weight_saver(
                self.model,
                self.hf_config,
                dtype=self.param_dtype,
                is_value_model=self.is_value_model,
                tie_word_embeddings=self.share_embeddings_and_output_weights,
            )

            torch.distributed.barrier()
            if self.rank == 0:
                hf_model_ckpt_path = get_hf_model_checkpoint_path(local_path)
                import warnings

                from accelerate import init_empty_weights

                with init_empty_weights(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if "mistral7b-rm" in self.config.model.path:
                        from transformers import MistralForSequenceClassification

                        model = MistralForSequenceClassification.from_pretrained(self.config.model.path)  # use score head instead of lm_head
                        state_dict["score.weight"] = state_dict["score.weight"]
                    else:
                        from transformers import AutoModelForCausalLM

                        model = AutoModelForCausalLM.from_pretrained(self.config.model.path, torch_dtype="auto")
                model.save_pretrained(hf_model_ckpt_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_model_ckpt_path)
                log_with_rank(f"Saved Huggingface config and tokenizer to {hf_model_ckpt_path}", rank=self.rank, logger=logger, log_only_rank_0=True)

                if hdfs_path is not None:
                    log_with_rank(f"Uploading checkpoint to {hdfs_path}", rank=self.rank, logger=logger, log_only_rank_0=True)
                    from verl.utils import hdfs_io

                    hdfs_io.makedirs(hdfs_path, exist_ok=True)
                    hdfs_io.copy(src=hf_model_ckpt_path, dst=hdfs_path, dirs_exist_ok=True)
                    log_with_rank(f"HDFS checkpoint uploaded to {hdfs_path}", rank=self.rank, logger=logger, log_only_rank_0=True)

        # Save Optimizer
        if self.should_save_optimizer:
            torch.distributed.barrier()

            optimizer_path = get_optimizer_checkpoint_path(local_path)
            self.optimizer.save_parameter_state(optimizer_path)
            log_with_rank(f"Saved optimizer state to {optimizer_path}", rank=self.rank, logger=logger)

        # Save RNG States
        if self.should_save_extra:
            torch.distributed.barrier()

            rng_state_path = get_rng_states_checkpoint_path(local_path, only_rank0_save=False)
            rng_state = self.get_rng_state()
            torch.save(rng_state, rng_state_path)
            log_with_rank(f"Saved rng states to {rng_state_path}", rank=self.rank, logger=logger)

            if self.rank == 0:
                optimizer_scheduler_path = get_optimizer_scheduler_checkpoint_path(local_path, only_rank0_save=True)
                if self.lr_scheduler is not None:
                    state_dict = self.lr_scheduler.state_dict()
                    torch.save(state_dict, optimizer_scheduler_path)
                    log_with_rank(f"Saved optimizer scheduler state to {optimizer_scheduler_path}", rank=self.rank, logger=logger, log_only_rank_0=True)

        self.previous_saved_paths.append(local_path)
