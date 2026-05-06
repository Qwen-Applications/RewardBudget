#!/bin/bash

# Unified evaluation script for GARO tasks
# Usage: ./recipe/GARO/scripts/eval.sh <model>

set -e

TASK_LIST="amc23,aime24,aime25,math500,OlympiadBench_physics_en_comp,OlympiadBench_math_en_comp" # omni_math
MODEL="$1"
DATA_DIR="/primus_datasets/zmy"
OUTPUT_DIR=${PRIMUS_OUTPUT_DIR}
TMP_DIR="/root/tmp"

# Validate inputs
if [ -z "$OUTPUT_DIR" ]; then
    echo "Error: PRIMUS_OUTPUT_DIR not set"
    exit 1
fi

# Support multiple tasks in one run (comma-separated).
# Split TASK_LIST into an array.
IFS=',' read -ra TASKS <<< "$TASK_LIST"

OUTPUT_PATH="${OUTPUT_DIR}/GARO_garo/"

DATA_PATHS=()
for t in "${TASKS[@]}"; do
  DATA_PATHS+=("${DATA_DIR}/GARO/test_data/${t}.parquet")
done

CHECKPOINT_DIR=${PRIMUS_SOURCE_CHECKPOINT_DIR}/${PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME}
CHECKPOINT_PATH="${CHECKPOINT_DIR}/actor/huggingface"

# Create output directory if it doesn't exist
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TMP_DIR}"

export NNODES=${NNODES:-1}
export NUM_ACCELERATORS=${NUM_ACCELERATORS:-8}
export GENERATION_BACKEND=${GENERATION_BACKEND:-hf}
export VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
export VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}
export VLLM_TENSOR_MODEL_PARALLEL_SIZE=${VLLM_TENSOR_MODEL_PARALLEL_SIZE:-1}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-True}
export VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL:-True}
export HF_GEN_BATCH_SIZE=${HF_GEN_BATCH_SIZE:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[INFO] Running evaluation for task(s): ${TASKS[*]}, model: $MODEL"
echo "[INFO] Data path(s): ${DATA_PATHS[*]}"
echo "[INFO] Checkpoint path: ${CHECKPOINT_PATH}"
echo "[INFO] Output path: ${OUTPUT_PATH}"
echo "[INFO] Generation backend: ${GENERATION_BACKEND}"
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

export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export TORCH_SYMM_MEM_DISABLE_MULTICAST=1

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

build_paths_for_k() {
  local k="$1"
  TMP_PATHS=()
  OUTPUT_PARQUET_PATHS=()
  EVAL_PATHS=()
  EVAL_JSONL_PATHS=()

  for t in "${TASKS[@]}"; do
    TMP_PATHS+=("${TMP_DIR}/${t}_${MODEL}_k${k}.parquet")
    OUTPUT_PARQUET_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_k${k}.parquet")
    EVAL_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_k${k}.eval.json")
    EVAL_JSONL_PATHS+=("${OUTPUT_PATH}/${t}_${MODEL}_k${k}.eval.details.jsonl")
  done
}

run_generation_for_k() {
  local k="$1"
  local temperature="$2"

  build_paths_for_k "${k}"

  echo "[INFO] Generating responses for k=${k}, temperature=${temperature}"
  echo "[INFO] Temp parquet path(s) for k=${k}: ${TMP_PATHS[*]}"
  echo "[INFO] Output parquet path(s) for k=${k}: ${OUTPUT_PARQUET_PATHS[*]}"

  # main_generation_torchrun_dp only accepts one data.path/data.output_path,
  # so generate each dataset separately under torchrun data parallelism.
  for idx in "${!TASKS[@]}"; do
    t=${TASKS[$idx]}
    data_path=${DATA_PATHS[$idx]}
    tmp=${TMP_PATHS[$idx]}

    echo "[INFO] Generating ${t} for k=${k}: ${data_path} -> ${tmp}"
    TORCHRUN_DP_RUN_ID="${t}_${MODEL}_k${k}" \
    torchrun --standalone --nproc_per_node=${NUM_ACCELERATORS} \
      -m verl.trainer.main_generation_torchrun_dp \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
      data.path="${data_path}" \
      data.output_path="${tmp}" \
      data.prompt_key=prompt \
      data.n_samples=${k} \
      data.batch_size=${HF_GEN_BATCH_SIZE} \
      model.path="${CHECKPOINT_PATH}" \
      +data.trust_remote_code=True \
      rollout.temperature=${temperature} \
      +rollout.backend=${GENERATION_BACKEND} \
      rollout.name=${GENERATION_BACKEND} \
      rollout.dtype=${VLLM_DTYPE} \
      rollout.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
      rollout.tensor_model_parallel_size=${VLLM_TENSOR_MODEL_PARALLEL_SIZE} \
      rollout.max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} \
      rollout.max_num_seqs=${VLLM_MAX_NUM_SEQS} \
      rollout.enforce_eager=${VLLM_ENFORCE_EAGER} \
      rollout.enable_chunked_prefill=${VLLM_ENABLE_CHUNKED_PREFILL} \
      +rollout.isolate_cuda_visible_devices=True \
      rollout.top_p=1.0 \
      rollout.top_k=-1 \
      rollout.prompt_length=1024 \
      rollout.response_length=8192 \
      rollout.mode=sync

    wait_for_file_ready "${tmp}"
  done
}

copy_outputs_for_k() {
  local k="$1"
  build_paths_for_k "${k}"

  for idx in "${!TASKS[@]}"; do
    t=${TASKS[$idx]}
    tmp=${TMP_PATHS[$idx]}
    out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}

    echo "[INFO] Copying generated parquet for ${t}, k=${k}: ${tmp} -> ${OUTPUT_PATH}"
    rclone copy "${tmp}" "${OUTPUT_PATH}" --progress --transfers=80
    wait_for_file_ready "${out_parquet}"
  done
}

run_evaluation_for_k() {
  local k="$1"
  build_paths_for_k "${k}"

  echo "[INFO] Evaluation path(s) for k=${k}: ${EVAL_PATHS[*]}"

  for idx in "${!TASKS[@]}"; do
    t=${TASKS[$idx]}
    out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}
    evalp=${EVAL_PATHS[$idx]}
    evalj=${EVAL_JSONL_PATHS[$idx]}

    echo "[INFO] Evaluating ${t}, k=${k}: ${out_parquet}"
    python3 recipe/GARO/scripts/evaluate/evaluation.py \
      --file "${out_parquet}" --k "${k}" \
      --output "${evalp}" \
      --output_jsonl "${evalj}" \
      --qwen_ip "${QWEN_IP}"

    echo "[INFO] Evaluation completed for ${t}, k=${k}. Results saved to ${evalp} (details: ${evalj})"
  done
}

# Generate/evaluate k=8 and k=1 into separate files to avoid overwrites.
run_generation_for_k 8 0.7
copy_outputs_for_k 8
run_evaluation_for_k 8

run_generation_for_k 1 0.0
copy_outputs_for_k 1
run_evaluation_for_k 1
