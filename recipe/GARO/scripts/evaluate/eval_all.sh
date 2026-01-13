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

TASK="math500,amc23,aime24,aime25,omni_math,minerva,OlympiadBench_math_en_comp,OlympiadBench_physics_en_comp"
MODEL="grpo"
STEP=300
JOB_ID="primusad7f74c10a9b94a3b69982f24z"
DATA_DIR="/primus_datasets/zmy"
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


# Support multiple tasks in one run (comma-separated), to avoid re-loading ckpt.
# Example:
#   ./eval.sh "math500,gsm8k" qwen 525 <job_id>
IFS=',' read -r -a TASKS <<< "${TASK}"

OUTPUT_PATH="${OUTPUT_DIR}/GARO/"

DATA_PATHS=()
TMP_PATHS=()
EVAL_PATHS=()
for t in "${TASKS[@]}"; do
  DATA_PATHS+=("${DATA_DIR}/GARO/test_data/${t}.parquet")
  TMP_PATHS+=("${TMP_DIR}/${t}_${MODEL}_${STEP}.parquet")
  EVAL_PATHS+=("${OUTPUT_DIR}/GARO/${t}_${MODEL}_${STEP}.eval.json")
done

# Replace JOB_ID in ${PRIMUS_SAVE_CHECKPOINT_DIR}
CHECKPOINT_DIR="/primus_oss_root/repository/quark_sft/${JOB_ID}/save_checkpoint"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_${STEP}/actor/huggingface"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}/GARO"
mkdir -p "${TMP_DIR}"

echo "[INFO] Running evaluation for task(s): ${TASKS[*]}, model: $MODEL, step: $STEP, job: $JOB_ID"
echo "[INFO] Data path(s): ${DATA_PATHS[*]}"
echo "[INFO] Checkpoint path: ${CHECKPOINT_PATH}"
echo "[INFO] Output path: ${OUTPUT_PATH} (and temp path(s): ${TMP_PATHS[*]})"
echo "[INFO] Evaluation path(s): ${EVAL_PATHS[*]}"

# =========================
# new-changes: print first row for debug (first task only)
python - <<PY
import pyarrow.parquet as pq

path = "${DATA_PATHS[0]}"
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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1


# Generate responses (load ckpt ONCE, then run all datasets sequentially)
# Build Hydra list args: data.paths=["p1","p2"], data.output_paths=["o1","o2"]
paths_arg="["
outs_arg="["
for p in "${DATA_PATHS[@]}"; do paths_arg+="\"${p}\","; done
for o in "${TMP_PATHS[@]}"; do outs_arg+="\"${o}\","; done
paths_arg=${paths_arg%,}
outs_arg=${outs_arg%,}
paths_arg+="]"
outs_arg+="]"

python -m verl.trainer.main_generation_multi \
  trainer.nnodes=${NNODES} \
  trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
  data.paths=${paths_arg} \
  data.output_paths=${outs_arg} \
  data.prompt_key=prompt \
  data.n_samples=1 \
  model.path=${CHECKPOINT_PATH} \
  +model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192 \
  +rollout.pipeline_model_parallel_size=1 \
  rollout.tensor_model_parallel_size=${NUM_ACCELERATORS} \
  +actor_rollout_ref.rollout.mode=async

# Copy results to output directory using rclone + evaluate each dataset
for idx in "${!TASKS[@]}"; do
  t=${TASKS[$idx]}
  tmp=${TMP_PATHS[$idx]}
  evalp=${EVAL_PATHS[$idx]}

  rclone copy ${tmp} ${OUTPUT_PATH} --progress --transfers=80

  python3 recipe/GARO/scripts/evaluate/evaluation.py \
    --file ${tmp} --k 1 \
    --output ${evalp}

  echo "[INFO] Evaluation completed for ${t}. Results saved to ${evalp}"
done