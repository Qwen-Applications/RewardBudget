#!/bin/bash

# Judge-only evaluation script for GARO tasks
# Usage: ./recipe/GARO/scripts/evaluate/eval_only.sh <model> <global_step> <job_id> [k]
#
# Assumption:
#   verl.trainer.main_generation_multi has finished already.
#   The generated parquet results are stored at:
#     /primus_oss_root/repository/quark_sft/${JOB_ID}/output/GARO/
#   with filenames:
#     ${task}_${model}_${global_step}.parquet

set -e

TASK_LIST="OlympiadBench_math_en_comp,OlympiadBench_physics_en_comp"
MODEL="$1"
STEP="$2"
JOB_ID="$3"
K="${4:-1}"  # default k=1, consistent with eval_all.sh; pass 16 if you want pass@16

# Where the generation outputs already exist
OUTPUT_PATH="/primus_oss_root/repository/quark_sft/${JOB_ID}/output/GARO"

# Split TASK_LIST into an array
IFS=',' read -ra TASKS <<< "$TASK_LIST"

echo "[INFO] Running JUDGE ONLY for task(s): ${TASKS[*]}, model: $MODEL, step: $STEP, job: $JOB_ID"
echo "[INFO] Input parquet dir: ${OUTPUT_PATH}"
echo "[INFO] k = ${K}"

for t in "${TASKS[@]}"; do
  parquet="${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.parquet"
  evalp="${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.eval_p${K}.json"
  evalj="${OUTPUT_PATH}/${t}_${MODEL}_${STEP}.eval_p${K}.details.jsonl"

  if [ ! -f "${parquet}" ]; then
    echo "[WARN] Missing parquet: ${parquet}  (skip ${t})"
    continue
  fi

  echo "[INFO] Judging ${t} ..."
  python3 recipe/GARO/scripts/evaluate/evaluation.py \
    --file "${parquet}" --k "${K}" \
    --output "${evalp}" \
    --output_jsonl "${evalj}"

  echo "[INFO] Done ${t}. Results: ${evalp} (details: ${evalj})"
done