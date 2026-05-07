#!/bin/bash

# Unified evaluation script for GARO tasks
# Usage: ./recipe/GARO/scripts/eval.sh <model>

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

wandb login REDACTED_WANDB_KEY

TASK_LIST="amc23,aime24,aime25,OlympiadBench_physics_en_comp" # OlympiadBench_math_en_comp, omni_math
MODEL="${1:-${MODEL:-}}"

if [ -z "${MODEL}" ]; then
    echo "Usage: $0 <model>"
    exit 1
fi

DATA_DIR="/primus_datasets/zmy"
OUTPUT_DIR=${PRIMUS_OUTPUT_DIR}
TMP_DIR=${PRIMUS_OUTPUT_DIR}

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
export HF_GEN_BATCH_SIZE=${HF_GEN_BATCH_SIZE:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# W&B parquet artifact upload. Set WANDB_PROJECT to enable by default.
export WANDB_PROJECT=${WANDB_PROJECT:-rb_generations}
export WANDB_PARQUET_UPLOAD=${WANDB_PARQUET_UPLOAD:-${WANDB_PROJECT:+1}}
export WANDB_PARQUET_UPLOAD=${WANDB_PARQUET_UPLOAD:-0}
export WANDB_GARO_RUN_ID=${WANDB_GARO_RUN_ID:-garo-eval-${MODEL}-${PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME:-unknown}-$(date +%Y%m%d%H%M%S)-$$}
export WANDB_GARO_RUN_NAME="${WANDB_GARO_RUN_NAME:-GARO eval ${MODEL} ${PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME:-unknown}}"
export WANDB_GARO_ARTIFACT_TYPE=${WANDB_GARO_ARTIFACT_TYPE:-garo-generation-parquet}

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

upload_parquet_to_wandb() {
  local file_path="$1"
  local task="$2"
  local k="$3"

  if [ "${WANDB_PARQUET_UPLOAD}" != "1" ]; then
    echo "[INFO] WANDB_PARQUET_UPLOAD=${WANDB_PARQUET_UPLOAD}; skip W&B upload for ${file_path}"
    return 0
  fi
  if [ -z "${WANDB_PROJECT:-}" ]; then
    echo "[ERROR] WANDB_PROJECT must be set when WANDB_PARQUET_UPLOAD=1"
    return 1
  fi

  echo "[INFO] Uploading parquet to W&B: ${file_path}"
  python3 - "${file_path}" "${task}" "${MODEL}" "${k}" "${CHECKPOINT_PATH}" <<'PY'
import os
import re
import sys

import wandb

file_path, task, model, k, checkpoint_path = sys.argv[1:6]

def sanitize(value, max_len=128):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return (value or "unknown")[:max_len]

project = os.environ["WANDB_PROJECT"]
entity = os.environ.get("WANDB_ENTITY") or None
run_id = sanitize(os.environ.get("WANDB_GARO_RUN_ID", "garo-eval"), max_len=64)
run_name = os.environ.get("WANDB_GARO_RUN_NAME") or f"GARO eval {model}"
artifact_type = os.environ.get("WANDB_GARO_ARTIFACT_TYPE", "garo-generation-parquet")
artifact_name = sanitize(f"garo-{task}-{model}-k{k}-parquet")

with wandb.init(
    project=project,
    entity=entity,
    id=run_id,
    name=run_name,
    resume="allow",
    job_type="garo-eval-upload",
    config={
        "model": model,
        "checkpoint_path": checkpoint_path,
        "generation_backend": os.environ.get("GENERATION_BACKEND"),
    },
) as run:
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata={
            "task": task,
            "model": model,
            "k": int(k),
            "checkpoint_path": checkpoint_path,
            "local_path": file_path,
            "num_bytes": os.path.getsize(file_path),
            "generation_backend": os.environ.get("GENERATION_BACKEND"),
        },
    )
    artifact.add_file(file_path, name=os.path.basename(file_path))
    run.log_artifact(artifact, aliases=["latest", f"k{k}", sanitize(task)])

print(f"[INFO] W&B artifact logged: {artifact_name}")
PY
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
  echo "[INFO] Shard root for k=${k}: ${TMP_DIR}"
  echo "[INFO] Output parquet path(s) for k=${k}: ${OUTPUT_PARQUET_PATHS[*]}"
  echo "[INFO] Local temp parquet path(s) for k=${k}: ${TMP_PATHS[*]}"

  # Build Hydra list args: data.paths=["p1","p2"], data.output_paths=["o1","o2"]
  # Write parquet locally first. Direct pyarrow/pandas writes to /primus_oss can leave 0-byte files.
  paths_arg="["
  outs_arg="["
  for p in "${DATA_PATHS[@]}"; do paths_arg+="\"${p}\","; done
  for o in "${TMP_PATHS[@]}"; do outs_arg+="\"${o}\","; done
  paths_arg=${paths_arg%,}
  outs_arg=${outs_arg%,}
  paths_arg+="]"
  outs_arg+="]"

  echo "[INFO] Generating all datasets for k=${k} in one torchrun."
  TORCHRUN_DP_RUN_ID="${MODEL}_k${k}_$(date +%s)" \
  TORCHRUN_DP_SHARD_ROOT="${TMP_DIR}" \
  torchrun --standalone --nproc_per_node=${NUM_ACCELERATORS} \
    -m verl.trainer.main_generation_torchrun_dp \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
    +data.paths=${paths_arg} \
    +data.output_paths=${outs_arg} \
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

  for idx in "${!OUTPUT_PARQUET_PATHS[@]}"; do
    tmp_parquet=${TMP_PATHS[$idx]}
    out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}
    t=${TASKS[$idx]}
    wait_for_file_ready "${tmp_parquet}"
    echo "[INFO] Copying parquet to output: ${tmp_parquet} -> ${OUTPUT_PATH}"
    rclone copy "${tmp_parquet}" "${OUTPUT_PATH}" --progress --transfers=80
    wait_for_file_ready "${out_parquet}"
    upload_parquet_to_wandb "${out_parquet}" "${t}" "${k}"
  done
}

run_evaluation_for_k() {
  local k="$1"
  build_paths_for_k "${k}"

  echo "[INFO] Evaluation path(s) for k=${k}: ${EVAL_PATHS[*]}"

  for idx in "${!TASKS[@]}"; do
    t=${TASKS[$idx]}
    out_parquet=${OUTPUT_PARQUET_PATHS[$idx]}
    tmp_parquet=${TMP_PATHS[$idx]}
    evalp=${EVAL_PATHS[$idx]}
    evalj=${EVAL_JSONL_PATHS[$idx]}

    echo "[INFO] Evaluating ${t}, k=${k}: ${tmp_parquet}"
    python3 recipe/GARO/scripts/evaluate/evaluation.py \
      --file "${tmp_parquet}" --k "${k}" \
      --output "${evalp}" \
      --output_jsonl "${evalj}" \
      --qwen_ip "${QWEN_IP}"

    echo "[INFO] Evaluation completed for ${t}, k=${k}. Results saved to ${evalp} (details: ${evalj})"
  done
}

# Generate/evaluate k=8 and k=1 into separate files to avoid overwrites.
run_generation_for_k 8 0.7
run_evaluation_for_k 8

run_generation_for_k 1 0.0
run_evaluation_for_k 1
