#!/bin/bash

# Evaluate existing GARO generation parquet files only.
# First enrich all generation parquet files with ground truth, then evaluate.
#
# Usage:
#   bash recipe/GARO/scripts/evaluate/eval_local.sh <model_name> <k>
#
# Example:
#   bash recipe/GARO/scripts/evaluate/eval_local.sh dapo_llama_1500 8

set -euo pipefail

# This script is assumed to be located at:
#   recipe/GARO/scripts/evaluate/eval_local.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TASK_LIST="amc23,aime24,aime25,math500,OlympiadBench_physics_en_comp"

MODEL_NAME="${1:-${MODEL_NAME:-dapo_llama_1500}}"
K="${2:-${K:-8}}"

GEN_DIR="${GEN_DIR:-recipe/GARO/generations}"
TEST_DATA_DIR="${TEST_DATA_DIR:-recipe/GARO/test_data}"
EVAL_DIR="${EVAL_DIR:-${GEN_DIR}/eval_results}"
ENRICHED_DIR="${ENRICHED_DIR:-${GEN_DIR}/eval_inputs_with_gt}"

mkdir -p "${EVAL_DIR}"
mkdir -p "${ENRICHED_DIR}"

IFS=',' read -ra TASKS <<< "${TASK_LIST}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] k: ${K}"
echo "[INFO] Generation dir: ${GEN_DIR}"
echo "[INFO] Test data dir: ${TEST_DATA_DIR}"
echo "[INFO] Enriched input dir: ${ENRICHED_DIR}"
echo "[INFO] Eval output dir: ${EVAL_DIR}"
echo "[INFO] Tasks: ${TASKS[*]}"

merge_gt_for_task() {
  local task="$1"

  local input_file="${GEN_DIR}/${task}_${MODEL_NAME}_k${K}.parquet"
  local test_jsonl="${TEST_DATA_DIR}/${task}.jsonl"
  local enriched_file="${ENRICHED_DIR}/${task}_${MODEL_NAME}_k${K}.with_gt.parquet"

  if [ ! -f "${input_file}" ]; then
    echo "[ERROR] Missing generation file: ${input_file}"
    exit 1
  fi

  if [ ! -f "${test_jsonl}" ]; then
    echo "[ERROR] Missing test jsonl file: ${test_jsonl}"
    exit 1
  fi

  echo "[INFO] Preparing enriched parquet for task=${task}"
  echo "[INFO] Generation input: ${input_file}"
  echo "[INFO] Test jsonl: ${test_jsonl}"
  echo "[INFO] Enriched parquet: ${enriched_file}"

  "${PYTHON_BIN}" - "${input_file}" "${test_jsonl}" "${enriched_file}" "${task}" <<'PY'
import json
import re
import sys
from pathlib import Path

import pandas as pd

gen_path = Path(sys.argv[1])
test_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
task = sys.argv[4]

def extract_prompt(x):
    """
    Generation parquet prompt is usually a string.
    Test jsonl prompt is usually:
      [{"role": "user", "content": "..."}]
    This function converts both to comparable string prompts.
    """
    if isinstance(x, str):
        return x

    if isinstance(x, list):
        user_parts = []
        all_parts = []
        for item in x:
            if isinstance(item, dict):
                content = item.get("content", "")
                role = item.get("role", "")
                if content:
                    all_parts.append(str(content))
                    if role == "user":
                        user_parts.append(str(content))
        if user_parts:
            return "\n".join(user_parts)
        if all_parts:
            return "\n".join(all_parts)
        return ""

    if isinstance(x, dict):
        return str(x.get("content", ""))

    return str(x)

def norm_prompt(s):
    s = extract_prompt(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_ground_truth(obj):
    rm = obj.get("reward_model", {})
    if isinstance(rm, dict):
        if "ground_truth" in rm:
            return rm["ground_truth"]
        if "groundtruth" in rm:
            return rm["groundtruth"]
        if "answer" in rm:
            return rm["answer"]

    for key in ["ground_truth", "groundtruth", "answer", "target"]:
        if key in obj:
            return obj[key]

    raise KeyError(f"Cannot find ground truth field in jsonl row keys={list(obj.keys())}")

gen_df = pd.read_parquet(gen_path)

if "prompt" not in gen_df.columns:
    raise ValueError(f"Generation parquet must contain column 'prompt'. Found columns: {list(gen_df.columns)}")

if "responses" not in gen_df.columns:
    raise ValueError(f"Generation parquet must contain column 'responses'. Found columns: {list(gen_df.columns)}")

test_rows = []
with open(test_path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        prompt = extract_prompt(obj.get("prompt", ""))
        gt = extract_ground_truth(obj)
        data_source = obj.get("data_source", task)

        test_rows.append(
            {
                "prompt": prompt,
                "_norm_prompt": norm_prompt(prompt),
                "ground_truth": gt,
                "data_source": data_source,
                "reward_model": obj.get("reward_model", {"ground_truth": gt}),
            }
        )

test_df = pd.DataFrame(test_rows)

if len(test_df) == 0:
    raise ValueError(f"Empty test jsonl: {test_path}")

gen_df = gen_df.copy()
gen_df["_norm_prompt"] = gen_df["prompt"].map(norm_prompt)

gt_map = {}
ds_map = {}
rm_map = {}
duplicate_prompts = set()

for row in test_df.to_dict("records"):
    key = row["_norm_prompt"]
    if key in gt_map:
        duplicate_prompts.add(key)

    gt_map[key] = row["ground_truth"]
    ds_map[key] = row["data_source"]
    rm_map[key] = row["reward_model"]

if duplicate_prompts:
    print(
        f"[WARN] Found {len(duplicate_prompts)} duplicate normalized prompts in test data. "
        f"Matching may use last occurrence."
    )

gen_df["ground_truth"] = gen_df["_norm_prompt"].map(gt_map)
gen_df["groundtruth"] = gen_df["ground_truth"]
gen_df["groundtruths"] = gen_df["ground_truth"]
gen_df["answer"] = gen_df["ground_truth"]
gen_df["data_source"] = gen_df["_norm_prompt"].map(ds_map).fillna(task)
gen_df["reward_model"] = gen_df["_norm_prompt"].map(rm_map)

missing_mask = gen_df["ground_truth"].isna()
missing_count = int(missing_mask.sum())

if missing_count > 0:
    if len(gen_df) == len(test_df):
        print(
            f"[WARN] {missing_count}/{len(gen_df)} rows failed prompt-based ground truth matching. "
            f"Falling back to row-order matching because lengths match."
        )

        gen_df["ground_truth"] = test_df["ground_truth"].tolist()
        gen_df["groundtruth"] = gen_df["ground_truth"]
        gen_df["groundtruths"] = gen_df["ground_truth"]
        gen_df["answer"] = gen_df["ground_truth"]
        gen_df["data_source"] = test_df["data_source"].tolist()
        gen_df["reward_model"] = test_df["reward_model"].tolist()
    else:
        examples = gen_df.loc[missing_mask, "prompt"].head(3).tolist()
        raise ValueError(
            f"Failed to match ground truth for {missing_count}/{len(gen_df)} rows, "
            f"and cannot fall back by row order because generation rows={len(gen_df)}, "
            f"test rows={len(test_df)}. Examples: {examples}"
        )

gen_df = gen_df.drop(columns=["_norm_prompt"])

out_path.parent.mkdir(parents=True, exist_ok=True)
gen_df.to_parquet(out_path, index=False)

print(f"[INFO] Wrote enriched parquet: {out_path}")
print(f"[INFO] Rows: {len(gen_df)}")
print(f"[INFO] Columns: {list(gen_df.columns)}")
print("[INFO] First ground_truth:", gen_df["ground_truth"].iloc[0])
PY
}

evaluate_task() {
  local task="$1"

  local enriched_file="${ENRICHED_DIR}/${task}_${MODEL_NAME}_k${K}.with_gt.parquet"
  local output_json="${EVAL_DIR}/${task}_${MODEL_NAME}_k${K}.eval.json"
  local output_jsonl="${EVAL_DIR}/${task}_${MODEL_NAME}_k${K}.eval.details.jsonl"

  if [ ! -f "${enriched_file}" ]; then
    echo "[ERROR] Missing enriched parquet file: ${enriched_file}"
    exit 1
  fi

  echo "[INFO] Evaluating task=${task}"
  echo "[INFO] Eval input: ${enriched_file}"
  echo "[INFO] Output: ${output_json}"
  echo "[INFO] Details: ${output_jsonl}"

  cmd=(
    "${PYTHON_BIN}" -m recipe.GARO.scripts.evaluate.evaluation
    --file "${enriched_file}"
    --k "${K}"
    --output "${output_json}"
    --output_jsonl "${output_jsonl}"
  )

  if [ -n "${QWEN_IP:-}" ]; then
    cmd+=(--qwen_ip "${QWEN_IP}")
  fi

  "${cmd[@]}"

  echo "[INFO] Done evaluating: ${task}"
}

echo "[INFO] ==============================="
echo "[INFO] Stage 1/2: merge ground truth"
echo "[INFO] ==============================="

for task in "${TASKS[@]}"; do
  merge_gt_for_task "${task}"
done

echo "[INFO] All enriched parquet files prepared."

echo "[INFO] ==============================="
echo "[INFO] Stage 2/2: evaluate"
echo "[INFO] ==============================="

for task in "${TASKS[@]}"; do
  evaluate_task "${task}"
done

echo "[INFO] All evaluations completed."