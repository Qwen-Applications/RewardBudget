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
JOB_ID=$4
DATA_DIR=${5:-"/primus_datasets/zmy"}
OUTPUT_DIR=${6:-${PRIMUS_OUTPUT_DIR}}
TMP_DIR="/root/tmp"

# Validate inputs
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: PRIMUS_OUTPUT_DIR not set and no output directory provided"
    exit 1
fi

if [ -z "$PRIMUS_SAVE_CHECKPOINT_DIR" ]; then
    echo "Error: PRIMUS_SAVE_CHECKPOINT_DIR not set"
    exit 1
fi


# Define data paths based on task
DATA_PATH="${DATA_DIR}/GARO/test_data/${TASK}.parquet"
TMP_PATH="${TMP_DIR}/${TASK}_${MODEL}_${STEP}.parquet"
OUTPUT_PATH="${OUTPUT_DIR}/GARO/"
EVAL_PATH=${OUTPUT_DIR}/GARO/${TASK}_${MODEL}_${STEP}.eval.json

# Replace JOB_ID in ${PRIMUS_SAVE_CHECKPOINT_DIR}
CHECKPOINT_DIR="/primus_oss_root/repository/quark_sft/${JOB_ID}/save_checkpoint"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_${STEP}/actor/huggingface"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/GARO"
mkdir -p "${TMP_DIR}"

echo "[INFO] Running evaluation for task: $TASK, model: $MODEL, step: $STEP, job: $JOB_ID"
echo "[INFO] Data path: $DATA_PATH"
echo "[INFO] Checkpoint path: ${CHECKPOINT_PATH}"
echo "[INFO] Output path: ${OUTPUT_PATH} (and temp path: ${TMP_PATH})"
echo "[INFO] Evaluation path: ${EVAL_PATH}"

# =========================
# new-changes: print first row for debug
python - <<PY
import pyarrow.parquet as pq

path = "${DATA_PATH}"
print("[DEBUG] Parquet path:", path)

pf = pq.ParquetFile(path)
print("[DEBUG] Schema:")
print(pf.schema)

# 读取前1行
table = pf.read_row_groups([0], columns=None).slice(0, 1)
rows = table.to_pylist()
if not rows:
    print("[DEBUG] Empty parquet!")
else:
    row = rows[0]
    print("[DEBUG] First row keys:", list(row.keys()))
    print("[DEBUG] First row prompt type:", type(row.get("prompt")).__name__)
    print("[DEBUG] First row prompt type:", type(row.get("prompt")).__name__)
    print("[DEBUG] First row prompt value:", row.get("prompt"))
PY
# end new-changes
# =========================

# export NNODES=1
# export NUM_ACCELERATORS=1
# export CUDA_VISIBLE_DEVICES=0
# export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export HYDRA_FULL_ERROR=1

# Generate responses
python -m verl.trainer.main_generation \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
  data.path=${DATA_PATH} \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${TMP_PATH} \
  model.path=${CHECKPOINT_PATH} \
  +model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192 \
  +rollout.pipeline_model_parallel_size=1 \
  rollout.tensor_model_parallel_size=${NUM_ACCELERATORS} \
  +actor_rollout_ref.rollout.mode=async

# Copy results to output directory using rclone
rclone copy ${TMP_PATH} ${OUTPUT_PATH} --progress --transfers=80

# Evaluate results
python3 recipe/GARO/scripts/evaluate/evaluation.py \
  --file ${TMP_PATH} --k 1 \
  --output ${EVAL_PATH}

echo "[INFO] Evaluation completed. Results saved to ${EVAL_PATH}"
