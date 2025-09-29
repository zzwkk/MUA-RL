#!/usr/bin/env bash
set -euxo pipefail

# TORCH DEBUG
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,P2P,NET,GRAPH,ENV,DYNDBG
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_RETRY_CNT=15

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:512'
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export VLLM_USE_V1=1

export HCCL_CONNECT_TIMEOUT=3600
export GLOO_SOCKET_TIMEOUT=3600
export PATH=$PATH:~/.local/bin \
&& rm -rf ./verl.egg-info \
&& pip3 install tensorboard 
&& pip3 install -e .
&& pip3 install -r ./requirements_sglang.txt
&& pip3 install transformers==4.51.1

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
MODEL_NAME=Qwen3-mua-grpo

############################# Parameter ##############################
# train
N_NODE=4
BATCH_SIZE=32
MINI_BATCH_SIZE=32
ROLLOUT_N=8
EPOCH_NUM=30
TEMPERATURE=1.0
MODEL_PATH="/path/to/model"

# reward&log
ENABLE_THINKING="nothink" # think / nothink
TRAINSET_VERSION="retail_airline"
REWARD="naive"
REWARD_FUNC="compute_score"
TRAIN_NAME="09251109"

# user model
CHAT_MODEL="gpt-4o-2024-11-20"
API_KEY=""  # OpenAI API Key
BASE_URL="" # OpenAI Base URL
#######################################################################

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

# Ray
CURRPWD="/workdir"
WORKING_DIR=${WORKING_DIR:-"${CURRPWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"

timestamp=$(date +%s)

# Launch the master node of Ray in the container
PET_MASTER_PORT=8277

#    trainer.max_actor_ckpt_to_keep=1 \
#    trainer.max_critic_ckpt_to_keep=1 \

if [ "$rank" -eq 0 ]; then
    ray start --head --port=$PET_MASTER_PORT  --min-worker-port=10002 --max-worker-port=10101
    # Wait for other nodes to register with the cluster
    sleep 10

    # Check the number of cluster nodes, wait for all nodes to join
    echo "Waiting for all nodes to join the Ray cluster..."
    start_time=$(date +%s)
    timeout=300  # 5 minutes timeout

    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))

        # Get the current number of active nodes
        active_nodes=$(ray status | sed -n '/Active:/,/Pending:/p' | grep "1 node_" | wc -l)

        echo "Current number of active nodes: $active_nodes / $N_NODE"

        if [ "$active_nodes" -ge "$N_NODE" ]; then
            echo "All nodes have joined the cluster!"
            break
        fi

        if [ "$elapsed" -ge "$timeout" ]; then
            echo "Error: Timeout while waiting for nodes to join the cluster (5 minutes), current node count: $active_nodes / $N_NODE"
            echo "Expected node count: $N_NODE, actual node count: $active_nodes"
            echo "Insufficient cluster nodes, training task cannot continue!"
            exit 1
        fi

        echo "Waiting for more nodes to join... (waited ${elapsed}s / ${timeout}s)"
        sleep 10
    done


    ray status

    ray job submit \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- \
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
    actor_rollout_ref.actor.fsdp_config.fsdp_size=32 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.response_length_one_turn=1024 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.model.enable_activation_offload=True\
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.validation_data_dir="$VALID_LOG_PATH"\
    trainer.logger=['console','tensorboard'] \
    trainer.rollout_data_dir="$ROLLOUT_LOG_PATH" \
    trainer.project_name=$MODEL_NAME \
    trainer.experiment_name=$MODEL_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$N_NODE \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/taubench_tool_config.yaml" \
    trainer.total_epochs=$EPOCH_NUM \
    trainer.default_local_dir=$CKPT_DIR \
    actor_rollout_ref.rollout.enable_thinking=${ENABLE_THINKING_BOOL} \
    actor_rollout_ref.rollout.multi_turn.enable_tokenization_sanity_check=False \
    # reward_model.reward_manager=$REWARD \
    # custom_reward_function.path=verl/utils/reward_score/taubench.py \
    # custom_reward_function.name=$REWARD_FUNC

    sleep 15
else
  # Waiting for main node to finish
  sleep 15
  ray start --address="$main:$PET_MASTER_PORT"  --min-worker-port=10002 --max-worker-port=10101
  echo "ray start worker join master $main:$PET_MASTER_PORT"
  while [ ! -f ${TENSORBOARD_DIR}/connection/log/main_done_${main}.txt ]; do
    echo "Waiting for main node to finish..."
    sleep 600
  done
fi