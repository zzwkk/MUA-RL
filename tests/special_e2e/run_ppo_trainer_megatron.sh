#!/usr/bin/env bash
set -xeuo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export VERL_LOGGING_LEVEL=INFO
export VERL_PPO_LOGGING_LEVEL=INFO

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen2.5-0.5B}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/${MODEL_ID}}
huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

USE_DUMMY_MODEL=${USE_DUMMY_MODEL:-False}
DUMMY_MODEL_PATH=${DUMMY_MODEL_PATH:-${HOME}/dummy_models/${MODEL_ID}}
if [ "$USE_DUMMY_MODEL" = "True" ]; then
    if [ -z "${DUMMY_MODEL_CONFIG_PATH}"  ]; then
        echo "[ERROR] DUMMY_MODEL_CONFIG_PATH not set"
        exit 1
    fi
    
    python scripts/init_random_model.py \
        --hf_model_path "${MODEL_PATH}" \
        --new_config_path "${DUMMY_MODEL_CONFIG_PATH}" \
        --output_path "${DUMMY_MODEL_PATH}"

    MODEL_PATH="${DUMMY_MODEL_PATH}"
fi

TRAIN_FILES=${TRAIN_FILES:-${HOME}/data/gsm8k/train.parquet}
VAL_FILES=${VAL_FILES:-${HOME}/data/gsm8k/test.parquet}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-gae}
# Validation
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
TEST_FREQ=${TEST_FREQ:--1}
# Save & Resume
RESUME_MODE=${RESUME_MODE:-disable}
SAVE_FREQ=${SAVE_FREQ:--1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN:-2400}
forward_max_token_len_per_gpu=${FWD_MAX_TOKEN_LEN:-4800}
train_traj_micro_bsz_per_gpu=${MICRO_BSZ:-2} # b
n_resp_per_prompt=4 # g

train_traj_micro_bsz=$((train_traj_micro_bsz_per_gpu * NUM_GPUS)) # b * n
train_traj_mini_bsz=$((train_traj_micro_bsz * 2)) # 2 * b * n
train_prompt_mini_bsz=$((train_traj_mini_bsz * n_resp_per_prompt)) # 2 * b * n / g
train_prompt_bsz=$((train_prompt_mini_bsz * 2)) # 4 * b * n / g

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

COMMON_PP=${COMMON_PP:-2}
COMMON_VPP=${COMMON_VPP:-2}
COMMON_CP=${COMMON_CP:-2}
COMMON_TP=${COMMON_TP:-2}
COMMON_EP=${COMMON_EP:-1}
COMMON_ETP=${COMMON_ETP:-null}

TRAIN_TP=${TRAIN_TP:-$COMMON_TP}
INFER_TP=${INFER_TP:-$COMMON_TP}

ACTOR_PP=${ACTOR_PP:-$COMMON_PP}
ACTOR_VPP=${ACTOR_VPP:-$COMMON_VPP}
ACTOR_CP=${ACTOR_CP:-$COMMON_CP}
ACTOR_TP=${ACTOR_TP:-$TRAIN_TP}
ACTOR_EP=${ACTOR_EP:-$COMMON_EP}
ACTOR_ETP=${ACTOR_ETP:-$COMMON_ETP}
ROLLOUT_TP=${ROLLOUT_TP:-$INFER_TP}
REF_PP=${REF_PP:-$COMMON_PP}
REF_VPP=${REF_VPP:-$COMMON_VPP}
REF_CP=${REF_CP:-$COMMON_CP}
REF_TP=${REF_TP:-$TRAIN_TP}
REF_EP=${REF_EP:-$COMMON_EP}
REF_ETP=${REF_ETP:-$COMMON_ETP}
CRITIC_PP=${CRITIC_PP:-$COMMON_PP}
CRITIC_VPP=${CRITIC_VPP:-$COMMON_VPP}
CRITIC_CP=${CRITIC_CP:-$COMMON_CP}
CRITIC_TP=${CRITIC_TP:-$TRAIN_TP}
CRITIC_EP=${CRITIC_EP:-$COMMON_EP}
CRITIC_ETP=${CRITIC_ETP:-$COMMON_ETP}
RM_PP=${RM_PP:-$COMMON_PP}
RM_VPP=${RM_VPP:-$COMMON_VPP}
RM_CP=${RM_CP:-$COMMON_CP}
RM_TP=${RM_TP:-$TRAIN_TP}
RM_EP=${RM_EP:-$COMMON_EP}
RM_ETP=${RM_ETP:-$COMMON_ETP}

ALL_OFFLOAD=${ALL_OFFLOAD:-False}
COMMON_PARAM_OFFLOAD=${COMMON_PARAM_OFFLOAD:-$ALL_OFFLOAD}
COMMON_GRAD_OFFLOAD=${COMMON_GRAD_OFFLOAD:-$ALL_OFFLOAD}
COMMON_OPTIMIZER_OFFLOAD=${COMMON_OPTIMIZER_OFFLOAD:-$ALL_OFFLOAD}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
ACTOR_GRAD_OFFLOAD=${ACTOR_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_PARAM_OFFLOAD=${CRITIC_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}
CRITIC_GRAD_OFFLOAD=${CRITIC_GRAD_OFFLOAD:-$COMMON_GRAD_OFFLOAD}
CRITIC_OPTIMIZER_OFFLOAD=${CRITIC_OPTIMIZER_OFFLOAD:-$COMMON_OPTIMIZER_OFFLOAD}
RM_PARAM_OFFLOAD=${RM_PARAM_OFFLOAD:-$COMMON_PARAM_OFFLOAD}

LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-null}

CHECKPOINT_CONTENTS=['model','hf_model','optimizer','extra']
SKIP_SAVE_HF_MODEL=${SKIP_SAVE_HF_MODEL:-0}
if [ $SKIP_SAVE_HF_MODEL -eq 1 ]; then
    CHECKPOINT_CONTENTS=['model','optimizer','extra']
fi

USE_DIST_CKPT=${USE_DIST_CKPT:-False}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-${HOME}/dist_ckpt/${MODEL_ID}}
if [ "$USE_DIST_CKPT" = "True" ]; then
    if [ "$USE_DUMMY_MODEL" = "True" ]; then
        DIST_CKPT_PATH=${HOME}/dist_ckpt_dummy/${MODEL_ID}
    fi
    python scripts/converter_hf_to_mcore.py \
        --hf_model_path "${MODEL_PATH}" \
        --output_path "${DIST_CKPT_PATH}"
fi

ENGINES=("vllm" "sglang_async")

exp_name="$(basename "${MODEL_ID,,}")-megatron-gsm8k-minimal"

for ENGINE in "${ENGINES[@]}"; do
    python3 -m verl.trainer.main_ppo --config-path=config \
        --config-name='ppo_megatron_trainer.yaml'\
        algorithm.adv_estimator="${ADV_ESTIMATOR}" \
        data.train_files="${TRAIN_FILES}" \
        data.val_files="${VAL_FILES}" \
        data.train_batch_size=${train_prompt_bsz} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$ACTOR_PP \
        actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=$ACTOR_VPP \
        actor_rollout_ref.actor.megatron.context_parallel_size=$ACTOR_CP \
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$ACTOR_TP \
        actor_rollout_ref.actor.megatron.expert_model_parallel_size=$ACTOR_EP \
        actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ACTOR_ETP \
        actor_rollout_ref.actor.megatron.param_offload=${ACTOR_PARAM_OFFLOAD} \
        actor_rollout_ref.actor.megatron.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
        actor_rollout_ref.actor.megatron.grad_offload=${ACTOR_GRAD_OFFLOAD} \
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
        actor_rollout_ref.actor.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.checkpoint.save_contents=$CHECKPOINT_CONTENTS \
        actor_rollout_ref.rollout.name="${ENGINE}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$REF_PP \
        actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=$REF_VPP \
        actor_rollout_ref.ref.megatron.context_parallel_size=$REF_CP \
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$REF_TP \
        actor_rollout_ref.ref.megatron.expert_model_parallel_size=$REF_EP \
        actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$REF_ETP \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD} \
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
        actor_rollout_ref.ref.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
        critic.optim.lr=2e-5 \
        critic.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
        critic.model.path="${MODEL_PATH}" \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        critic.ppo_max_token_len_per_gpu=${forward_max_token_len_per_gpu} \
        critic.megatron.pipeline_model_parallel_size=$CRITIC_PP \
        critic.megatron.virtual_pipeline_model_parallel_size=$CRITIC_VPP \
        critic.megatron.context_parallel_size=$CRITIC_CP \
        critic.megatron.tensor_model_parallel_size=$CRITIC_TP \
        critic.megatron.expert_model_parallel_size=$CRITIC_EP \
        critic.megatron.expert_tensor_parallel_size=$CRITIC_ETP \
        critic.megatron.param_offload=${CRITIC_PARAM_OFFLOAD} \
        critic.megatron.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD} \
        critic.megatron.grad_offload=${CRITIC_GRAD_OFFLOAD} \
        critic.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
        critic.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
        critic.checkpoint.save_contents=$CHECKPOINT_CONTENTS \
        reward_model.enable=True \
        reward_model.model.path="${MODEL_PATH}" \
        reward_model.micro_batch_size_per_gpu=${train_traj_micro_bsz_per_gpu} \
        reward_model.megatron.pipeline_model_parallel_size=$RM_PP \
        reward_model.megatron.virtual_pipeline_model_parallel_size=$RM_VPP \
        reward_model.megatron.context_parallel_size=$RM_CP \
        reward_model.megatron.tensor_model_parallel_size=$RM_TP \
        reward_model.megatron.expert_model_parallel_size=$RM_TP \
        reward_model.megatron.expert_tensor_parallel_size=$RM_TP \
        reward_model.megatron.param_offload=${RM_PARAM_OFFLOAD} \
        reward_model.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
        reward_model.megatron.dist_checkpointing_path=${DIST_CKPT_PATH} \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_penalty=kl \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='verl-test' \
        trainer.experiment_name="${exp_name}" \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
        trainer.test_freq="${TEST_FREQ}" \
        trainer.save_freq="${SAVE_FREQ}" \
        trainer.resume_mode="${RESUME_MODE}" \
        trainer.total_epochs=2 \
        trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" $@
done
