#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-${MODEL_PATH:-}}
DATA_FILE=${2:-${DATA_FILE:-}}
OUTPUT_DIR=${3:-${OUTPUT_DIR:-outputs/reward_budget_eval}}
EXTRA_ARGS=("${@:4}")

if [[ -z "${MODEL_PATH}" || -z "${DATA_FILE}" ]]; then
  echo "Usage: bash recipe/RewardBudget/scripts/evaluate/run_eval.sh <model_or_checkpoint> <eval.parquet> [output_dir]"
  exit 1
fi

K=${K:-1}
if ! [[ "${K}" =~ ^[1-9][0-9]*$ ]]; then
  echo "K must be a positive integer, got: ${K}" >&2
  exit 1
fi
# Greedy decoding for pass@1; sample independently for pass@k (k > 1).
if (( K == 1 )); then
  GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.0}
else
  GEN_TEMPERATURE=${GEN_TEMPERATURE:-1.0}
fi
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TP_SIZE=${TP_SIZE:-1}
PROMPT_KEY=${PROMPT_KEY:-prompt}
GEN_FILE="${OUTPUT_DIR}/generations.parquet"
METRICS_FILE="${OUTPUT_DIR}/eval.json"
DETAILS_FILE="${OUTPUT_DIR}/eval.details.jsonl"

mkdir -p "${OUTPUT_DIR}"

python3 -m verl.trainer.main_generation \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  data.path="${DATA_FILE}" \
  data.prompt_key="${PROMPT_KEY}" \
  data.n_samples="${K}" \
  data.output_path="${GEN_FILE}" \
  model.path="${MODEL_PATH}" \
  rollout.name=vllm \
  rollout.tensor_model_parallel_size="${TP_SIZE}" \
  rollout.temperature="${GEN_TEMPERATURE}" \
  rollout.top_p=1.0 \
  rollout.max_num_batched_tokens=8192 \
  "${EXTRA_ARGS[@]}"

python3 -m recipe.RewardBudget.evaluation \
  --file "${GEN_FILE}" \
  --k "${K}" \
  --output "${METRICS_FILE}" \
  --output_jsonl "${DETAILS_FILE}"
