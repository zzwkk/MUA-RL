# run on 1nodes 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

export PATH=$PATH:~/.local/bin \
&& rm -rf ./verl.egg-info \
&& pip3 install tensorboard \
&& pip3 install -e . \
&& pip3 install -r ./requirements_sglang.txt \
&& pip3 install transformers==4.51.1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
MODEL_NAME=Mua-8B

############################ Parameter ##############################
# train
BATCH_SIZE=8
MINI_BATCH_SIZE=8
ROLLOUT_N=16
EPOCH_NUM=30
TEMPERATURE=1.0
MODEL_PATH="/path/to/model"

# reward&log
ENABLE_THINKING="nothink"      # think/nothink
TRAINSET_VERSION="retail"
REWARD="naive"
REWARD_FUNC="compute_score"
TRAIN_NAME="test" # change it to your own training name


# user model
CHAT_MODEL="gpt-4o-2024-11-20"
API_KEY=""  # OpenAI API Key
BASE_URL="" # OpenAI Base URL
######################################################################

if [ "$ENABLE_THINKING" == "think" ]; then
    ENABLE_THINKING_BOOL="True"
else
    ENABLE_THINKING_BOOL="False"
fi

SUFFIX="b${BATCH_SIZE}_mb${MINI_BATCH_SIZE}_n${ROLLOUT_N}_${ENABLE_THINKING}_${TRAINSET_VERSION}_R${REWARD_FUNC}_T${TEMPERATURE}_${TRAIN_NAME}"

CKPT_DIR=/path/to/CKPT/${MODEL_NAME}-${SUFFIX}
export TENSORBOARD_DIR=/path/to/tensorboard/${MODEL_NAME}-${SUFFIX}
ROLLOUT_LOG_PATH=/path/to/log/${MODEL_NAME}-${SUFFIX}/rollout_log
VALID_LOG_PATH=/path/to/log/${MODEL_NAME}-${SUFFIX}/valid_log

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export CHAT_MODEL=$CHAT_MODEL
export API_KEY=$API_KEY
export BASE_URL=$BASE_URL
#: > output.log

retail_path=$PROJECT_DIR/data/retail_test.parquet
airline_path=$PROJECT_DIR/data/airline_test.parquet

retail_empty_output=$PROJECT_DIR/data/retail_test_empty_output.parquet
airline_empty_output=$PROJECT_DIR/data/airline_test_empty_output.parquet

train_files="['$retail_empty_output', '$airline_empty_output']"
test_files="['$retail_path', '$airline_path']"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='mua_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=5000 \
    data.max_response_length=27768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.response_length_one_turn=1024 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.rollout_data_dir="$ROLLOUT_LOG_PATH" \
    trainer.validation_data_dir="$VALID_LOG_PATH"\
    trainer.project_name=$MODEL_NAME \
    trainer.experiment_name=$MODEL_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/taubench_tool_config.yaml" \
    trainer.total_epochs=$EPOCH_NUM \
    trainer.default_local_dir=$CKPT_DIR \
    actor_rollout_ref.rollout.enable_thinking=${ENABLE_THINKING_BOOL} \
    # reward_model.reward_manager=$REWARD \
    # custom_reward_function.path=verl/utils/reward_score/taubench_v1.py \
    # custom_reward_function.name=$REWARD_FUNC