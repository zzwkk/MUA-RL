#!/bin/bash
set -x
ulimit -n 65535

BASE_DIR="/path/to/CKPT/"
MODEL_NAME="your_model_name-your_suffix" # MODEL_NAME-SUFFIX
PARALLEL=true  # true: parallel, false: serial
MAX_JOBS=2     # max jobs for parallel execution

# specify the step numbers to process, leave empty to process all steps; if the step does not exist ckpt, skip
SPECIFIC_STEPS=()  # for example: (10 20 30) or leave empty ()

# build the full path
MODEL_DIR="$BASE_DIR/$MODEL_NAME"

# check if the model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory $MODEL_DIR does not exist!"
    exit 1
fi

# find all global_step_* directories
GLOBAL_STEPS=()

if [ ${#SPECIFIC_STEPS[@]} -gt 0 ]; then
    # if specified specific steps, only process these steps
    echo "Specific steps configured: ${SPECIFIC_STEPS[*]}"
    for step_num in "${SPECIFIC_STEPS[@]}"; do
        step_dir="$MODEL_DIR/global_step_$step_num"
        if [ -d "$step_dir" ]; then
            # check if the directory contains actor subdirectory
            if [ -d "$step_dir/actor" ]; then
                GLOBAL_STEPS+=("$step_dir")
                echo "Found configured step: $step_num"
            else
                echo "Skipping step $step_num - actor directory not found in $step_dir"
            fi
        else
            echo "Skipping step $step_num - directory $step_dir does not exist"
        fi
    done
else
    # if no specific steps specified, process all found steps
    echo "No specific steps configured, processing all available steps"
    for dir in $(ls -d $MODEL_DIR/global_step_* 2>/dev/null); do
        # check if the directory contains actor subdirectory
        if [ -d "$dir/actor" ]; then
            GLOBAL_STEPS+=("$dir")
        else
            echo "Skipping $dir - actor directory not found"
        fi
    done
fi

if [ ${#GLOBAL_STEPS[@]} -eq 0 ]; then
    if [ ${#SPECIFIC_STEPS[@]} -gt 0 ]; then
        echo "No valid requested steps found in $MODEL_DIR"
    else
        echo "No valid global_step_* directories with actor subdirectory found in $MODEL_DIR"
    fi
    exit 0
fi

# process each global_step directory
process_step() {
    local global_step_dir=$1
    local step_num=$(basename $global_step_dir | grep -oP 'global_step_\K\d+')
    local iter_dir="${MODEL_DIR}/iter_$(printf "%08d" $step_num)/actor/unify_checkpoint"

    if [ ! -d "$iter_dir" ]; then
        local local_dir="$global_step_dir/actor"
        local target_dir="$iter_dir"

        echo "Processing: $global_step_dir -> $iter_dir"

        # execute merge command
        python3 scripts/model_merger.py merge \
            --backend fsdp \
            --local_dir "$local_dir" \
            --target_dir "$target_dir"

        if [ $? -eq 0 ]; then
            echo "Successfully merged $global_step_dir to $iter_dir"
        else
            echo "Error merging $global_step_dir"
            return 1
        fi
    else
        echo "Skipping $global_step_dir - $iter_dir already exists"
    fi
}

# execute according to parallel/serial setting
if [ "$PARALLEL" = true ]; then
    echo "Running in parallel mode (max jobs: $MAX_JOBS)"
    for step in "${GLOBAL_STEPS[@]}"; do
        # wait until there is a available parallel slot
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            sleep 1
        done

        # execute task in background
        process_step "$step" &
    done
    # wait for all background tasks to complete
    wait
else
    echo "Running in serial mode"
    for step in "${GLOBAL_STEPS[@]}"; do
        process_step "$step"
    done
fi

echo "All tasks completed"