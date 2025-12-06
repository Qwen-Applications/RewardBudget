#!/bin/bash

# Unified evaluation script for GARO tasks
# Usage: ./recipe/GARO/scripts/eval.sh <task> <global_step> [data_root] [output_dir]

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <task> <model> <global_step> [data_root] [output_dir]"
    echo "Example: $0 grpo math500 525"
    exit 1
fi

TASK=$1
MODEL=$2
STEP=$3
DATA_DIR=${4:-"/primus_datasets/zmy"}
OUTPUT_DIR=${5:-${PRIMUS_OUTPUT_DIR}}

# Define data paths based on task
DATA_PATH="${DATA_DIR}/GARO/test_data/${TASK}.parquet"
OUTPUT_PATH="${OUTPUT_DIR}/GARO/${TASK}_${MODEL}_${STEP}.parquet"

# Validate inputs
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: PRIMUS_OUTPUT_DIR not set and no output directory provided"
    exit 1
fi

if [ -z "$PRIMUS_SAVE_CHECKPOINT_DIR" ]; then
    echo "Error: PRIMUS_SAVE_CHECKPOINT_DIR not set"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/GARO"

echo "[INFO] Running evaluation for task: $TASK, model: $MODEL, step: $STEP"
echo "[INFO] Data path: $DATA_PATH"
echo "[INFO] Output path: ${OUTPUT_PATH}"

# Generate responses
python -m verl.trainer.main_generation \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
  data.path=${DATA_PATH} \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${OUTPUT_PATH} \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR}/global_step_${STEP} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

# Evaluate results
python3 recipe/GARO/evaluation.py \
  --file ${OUTPUT_PATH} --k 1 \
  --output ${OUTPUT_DIR}/GARO/eval_${MODEL}_${STEP}.json

echo "[INFO] Evaluation completed. Results saved to ${OUTPUT_DIR}/GARO/eval_${MODEL}_${STEP}.json"
