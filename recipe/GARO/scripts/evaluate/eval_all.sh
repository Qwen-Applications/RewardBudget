#!/bin/bash

# Unified evaluation script for GARO tasks
# Usage: ./recipe/GARO/scripts/eval.sh <model> <global_step> <job_id>

set -e

# Check arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model> <global_step> <job_id>"
    echo "Example: $0 grpo 300 primusad7f74c10a9b94a3b69982f24z"
    exit 1
fi

TASK_LIST="math500,amc23,aime24,aime25,omni_math,OlympiadBench_math_en_comp,OlympiadBench_physics_en_comp"
MODEL="$1"
STEP="$2"
JOB_ID="$3"
DATA_DIR="/primus_datasets/zmy"
OUTPUT_DIR=${PRIMUS_OUTPUT_DIR}
TMP_DIR="/root/tmp"

# Validate inputs
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: PRIMUS_OUTPUT_DIR not set"
    exit 1
fi

if [ -z "$PRIMUS_SAVE_CHECKPOINT_DIR" ]; then
    echo "Error: PRIMUS_SAVE_CHECKPOINT_DIR not set"
    exit 1
fi

# Support multiple tasks in one run (comma-separated), to avoid re-loading ckpt.
# Split TASK_LIST into an array
IFS=',' read -ra TASKS <<< "$TASK_LIST"

OUTPUT_PATH="${OUTPUT_DIR}/GARO_garo/"

DATA_PATHS=()
TMP_PATHS=()
OUTPUT_PARQUET_PATHS=()
EVAL_PATHS=()
EVAL_JSONL_PATHS=()
for t in "${TASKS[@]}"; do
  DATA_PATHS+=("${DATA_DIR}/GARO/test_data/${t}.parquet")
  TMP_PATHS+=("${TMP_DIR}/${t}_${MODEL}_${STEP}.parquet")
  OUTPUT_PARQUET_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.parquet")
  EVAL_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.eval.json")
  EVAL_JSONL_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.eval.details.jsonl")
done

# Replace JOB_ID in ${PRIMUS_SAVE_CHECKPOINT_DIR}
CHECKPOINT_DIR="/primus_oss_root/repository/quark_sft/${JOB_ID}/save_checkpoint"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/global_step_${STEP}/actor/huggingface"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TMP_DIR}"

echo "[INFO] Running evaluation for task(s): ${TASKS[*]}, model: $MODEL, step: $STEP, job: $JOB_ID"
echo "[INFO] Data path(s): ${DATA_PATHS[*]}"
echo "[INFO] Checkpoint path: ${CHECKPOINT_PATH}"
echo "[INFO] Output path: ${OUTPUT_PATH} (and temp path(s): ${TMP_PATHS[*]})"
echo "[INFO] Evaluation path(s): ${EVAL_PATHS[*]}"

# =========================
# print first row for debug (first task only)
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
    print("[DEBUG] First row prompt value:", row.get("prompt"))
PY
# =========================

export NNODES=1
export NUM_ACCELERATORS=16
# export CUDA_VISIBLE_DEVICES=$(seq -s , 0 $((NUM_ACCELERATORS-1)))
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
  +data.paths=${paths_arg} \
  +data.output_paths=${outs_arg} \
  data.prompt_key=prompt \
  data.n_samples=8 \
  model.path=${CHECKPOINT_PATH} \
  +model.trust_remote_code=True \
  rollout.temperature=0.7 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192 \
  rollout.data_parallel_size=1 \
  rollout.pipeline_model_parallel_size=1 \
  rollout.tensor_model_parallel_size=1 \
  rollout.mode=sync

echo "[INFO] Output parquet path(s): ${OUTPUT_PARQUET_PATHS[*]}"

wait_for_file_ready() {
  local file_path="$1"
  local max_tries="${2:-60}"
  local sleep_secs="${3:-2}"
  local prev_size="-1"
  local curr_size="-1"
  local stable_count=0

  for ((i=1; i<=max_tries; i++)); do
    if [ -s "${file_path}" ]; then
      curr_size=$(stat -c%s "${file_path}" 2>/dev/null || echo -1)
      if [ "${curr_size}" -gt 0 ] && [ "${curr_size}" -eq "${prev_size}" ]; then
        stable_count=$((stable_count + 1))
      else
        stable_count=0
      fi
      if [ "${stable_count}" -ge 1 ]; then
        echo "[INFO] File ready: ${file_path} (size=${curr_size})"
        return 0
      fi
      prev_size="${curr_size}"
    fi
    echo "[INFO] Waiting for file to be ready (${i}/${max_tries}): ${file_path}"
    sleep "${sleep_secs}"
  done

  echo "[ERROR] File not ready after waiting: ${file_path}"
  return 1
}

# Phase 1: wait for all generated parquet files to be fully materialized.
for idx in "${!TASKS[@]}"; do
  t=${TASKS[$idx]}
  tmp=${TMP_PATHS[$idx]}
  echo "[INFO] Waiting for generated parquet of ${t}: ${tmp}"
  wait_for_file_ready "${tmp}"
done

# Phase 2: copy all parquet files to the output directory first.
for idx in "${!TASKS[@]}"; do
  t=${TASKS[$idx]}
  tmp=${TMP_PATHS[$idx]}
  out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}

  echo "[INFO] Copying generated parquet for ${t}: ${tmp} -> ${OUTPUT_PATH}"
  rclone copy ${tmp} ${OUTPUT_PATH} --progress --transfers=80
  wait_for_file_ready "${out_parquet}"
done

# Phase 3: after every parquet is present and stable, run evaluation.
for idx in "${!TASKS[@]}"; do
  t=${TASKS[$idx]}
  out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}
  evalp=${EVAL_PATHS[$idx]}
  evalj=${EVAL_JSONL_PATHS[$idx]}

  python3 recipe/GARO/scripts/evaluate/evaluation.py \
    --file ${out_parquet} --k 8 \
    --output ${evalp} \
    --output_jsonl ${evalj} \
    --qwen_ip "${QWEN_IP}"

  echo "[INFO] Evaluation completed for ${t}. Results saved to ${evalp} (details: ${evalj})"
done