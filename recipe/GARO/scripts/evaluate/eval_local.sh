#!/bin/bash

# Evaluate existing GARO generation parquet files only.
# First merge generation answers with ground truth, then evaluate.
#
# Usage:
#   bash recipe/GARO/scripts/evaluate/eval_local.sh <model_name> <k>
#
# Example:
#   bash recipe/GARO/scripts/evaluate/eval_local.sh drgrpo_llama 8

set -euo pipefail

# This script is assumed to be located at:
#   recipe/GARO/scripts/evaluate/eval_local.sh
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

TASK_LIST="${TASK_LIST:-amc23,aime24,aime25}"

MODEL_NAME="${1:-${MODEL_NAME:-klcov_qwen_800}}"
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

  echo "[INFO] Preparing merged parquet for task=${task}"
  echo "[INFO] Generation input: ${input_file}"
  echo "[INFO] Test jsonl: ${test_jsonl}"
  echo "[INFO] Merged parquet: ${enriched_file}"

  "${PYTHON_BIN}" - "${input_file}" "${test_jsonl}" "${enriched_file}" "${task}" "${K}" <<'PY'
import json
import re
import sys
from pathlib import Path

import pandas as pd

gen_path = Path(sys.argv[1])
test_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
task = sys.argv[4]
k = int(sys.argv[5])


def is_none_like(x):
    if x is None:
        return True

    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def extract_prompt(x):
    """
    Generation parquet prompt is usually a string.

    Test jsonl prompt is often:
      [{"role": "user", "content": "..."}]
    """
    if is_none_like(x):
        return ""

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
            else:
                all_parts.append(str(item))

        if user_parts:
            return "\n".join(user_parts)

        if all_parts:
            return "\n".join(all_parts)

        return ""

    if isinstance(x, dict):
        if "content" in x:
            return str(x.get("content", ""))

        if "prompt" in x:
            return extract_prompt(x.get("prompt", ""))

        return json.dumps(x, ensure_ascii=False, default=str)

    return str(x)


def norm_prompt(x):
    s = extract_prompt(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_ground_truth(obj):
    rm = obj.get("reward_model", {})

    if isinstance(rm, dict):
        for key in ["ground_truth", "groundtruth", "answer", "target", "gt"]:
            if key in rm:
                return rm[key]

    for key in ["ground_truth", "groundtruth", "answer", "target", "gt"]:
        if key in obj:
            return obj[key]

    raise KeyError(
        f"Cannot find ground truth field in jsonl row keys={list(obj.keys())}"
    )


def to_str(x):
    if is_none_like(x):
        return ""

    if isinstance(x, str):
        return x

    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, ensure_ascii=False, default=str)

    return str(x)


def response_to_text(x):
    """
    Convert one generated response object into plain text.

    Handles:
      - string
      - dict with response/text/content/output fields
      - list of chat messages
      - numpy/pyarrow arrays
      - one single response object
    """
    if is_none_like(x):
        return ""

    if isinstance(x, str):
        return x

    if not isinstance(x, (str, bytes)) and hasattr(x, "tolist"):
        try:
            return response_to_text(x.tolist())
        except Exception:
            pass

    if isinstance(x, dict):
        for key in [
            "response",
            "text",
            "content",
            "generation",
            "generated_text",
            "answer",
            "output",
            "message",
        ]:
            if key in x and x[key] is not None:
                return response_to_text(x[key])

        return json.dumps(x, ensure_ascii=False, default=str)

    if isinstance(x, list):
        parts = []

        for item in x:
            if isinstance(item, dict) and item.get("content") is not None:
                parts.append(str(item["content"]))
            else:
                parts.append(response_to_text(item))

        return "\n".join([p for p in parts if p is not None])

    return str(x)


def to_response_list(x):
    """
    Generation parquet column `responses` should usually be a list of k generations.

    This function also handles tuple, numpy arrays, pyarrow arrays, or a single string.
    """
    if is_none_like(x):
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, tuple):
        return list(x)

    if not isinstance(x, (str, bytes)) and hasattr(x, "tolist"):
        y = x.tolist()

        if isinstance(y, list):
            return y

        return [y]

    return [x]


def normalize_list(xs, target_k):
    xs = list(xs)

    if target_k <= 0:
        return xs

    if len(xs) < target_k:
        xs = xs + [""] * (target_k - len(xs))

    if len(xs) > target_k:
        xs = xs[:target_k]

    return xs


def clean_candidate_answer(s):
    if s is None:
        return ""

    s = str(s).strip()

    if not s:
        return ""

    s = s.replace("\u3000", " ")
    s = re.sub(r"^[:：=\-\s]+", "", s).strip()

    if not s:
        return ""

    # For fallback candidates, avoid swallowing a whole paragraph.
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if not lines:
        return ""

    s = lines[0].strip()

    if not s:
        return ""

    # Remove markdown emphasis.
    s = re.sub(r"^\*+", "", s)
    s = re.sub(r"\*+$", "", s).strip()

    if not s:
        return ""

    # Strip LaTeX inline math wrappers.
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()

    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    s = s.strip(" \t\n\r。.;,，")

    if not s:
        return ""

    # Avoid returning a whole sentence as an answer.
    if len(s) > 120:
        m = re.search(
            r"(-?\d+(?:\.\d+)?(?:/\d+)?|"
            r"\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|"
            r"[A-Za-z0-9_+\-*/^=]+)\s*[。.;,，]?$",
            s,
        )

        if m:
            s = m.group(1).strip()
        else:
            return ""

    return s.strip()


def extract_last_boxed(text):
    """
    Extract content from the last complete \\boxed{...}.

    Supports nested braces, e.g.:
      \\boxed{\\frac{1}{2}}

    Also supports simple forms like:
      \\boxed 42
    """
    text = response_to_text(text)
    token = r"\boxed"

    starts = []
    idx = 0

    while True:
        j = text.find(token, idx)

        if j == -1:
            break

        p = j + len(token)

        while p < len(text) and text[p].isspace():
            p += 1

        if p < len(text) and text[p] == "{":
            starts.append(p)

        idx = j + len(token)

    # Balanced-brace extraction for \boxed{...}
    for open_i in reversed(starts):
        depth = 0
        content_start = open_i + 1
        i = open_i

        while i < len(text):
            ch = text[i]

            # Do not treat escaped braces like \{ or \} as structural braces.
            if ch == "\\":
                i += 2
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    return clean_candidate_answer(text[content_start:i])

            i += 1

    # Simple non-nested fallback for \boxed{...}
    matches = re.findall(r"\\boxed\s*\{([^{}]*)\}", text, flags=re.DOTALL)

    if matches:
        return clean_candidate_answer(matches[-1])

    # Fallback for \boxed 42 or \boxed \frac{1}{2}
    matches = re.findall(
        r"\\boxed\s+(.{1,120})",
        text,
        flags=re.IGNORECASE,
    )

    if matches:
        return clean_candidate_answer(matches[-1])

    return ""


def extract_final_answer_fallback(text):
    """
    Fallback when no boxed answer exists.

    Looks from bottom to top for common final-answer markers.
    """
    text = response_to_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    marker_patterns = [
        r"(?:final\s+answer|final\s+result)\s*(?:is)?\s*[:：]?\s*(.+)$",
        r"(?:the\s+answer\s+is|answer\s+is)\s*[:：]?\s*(.+)$",
        r"(?:答案|最终答案|最后答案)\s*(?:是|为)?\s*[:：]?\s*(.+)$",
        r"(?:therefore|thus|so)\s*,?\s*(?:the\s+answer\s+is)?\s*[:：]?\s*(.+)$",
    ]

    for line in reversed(lines):
        for pat in marker_patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)

            if m:
                raw_cand = m.group(1) if m.lastindex else ""
                cand = clean_candidate_answer(raw_cand)

                if cand:
                    boxed = extract_last_boxed(cand)
                    return boxed or cand

    # Last-resort: use the last standalone numeric-looking expression.
    # This is intentionally conservative.
    numeric_matches = re.findall(
        r"(?<![A-Za-z])(-?\d+(?:\.\d+)?(?:/\d+)?)(?![A-Za-z])",
        text,
    )

    if numeric_matches:
        return clean_candidate_answer(numeric_matches[-1])

    return ""


def extract_one_answer(resp):
    text = response_to_text(resp)

    ans = extract_last_boxed(text)

    if ans:
        return ans

    ans = extract_final_answer_fallback(text)

    if ans:
        return ans

    return ""


def extract_answers(responses):
    response_list = to_response_list(responses)
    answers = [to_str(extract_one_answer(resp)) for resp in response_list]
    return normalize_list(answers, k)


def extract_response_texts(responses):
    response_list = to_response_list(responses)
    texts = [response_to_text(resp) for resp in response_list]
    return normalize_list(texts, k)


# =========================
# Load generation parquet
# =========================

gen_df = pd.read_parquet(gen_path)

if len(gen_df) == 0:
    raise ValueError(f"Empty generation parquet: {gen_path}")

if "prompt" not in gen_df.columns:
    raise ValueError(
        f"Generation parquet must contain column 'prompt'. "
        f"Found columns: {list(gen_df.columns)}"
    )

if "responses" not in gen_df.columns:
    raise ValueError(
        f"Generation parquet must contain column 'responses'. "
        f"Found columns: {list(gen_df.columns)}"
    )


# =========================
# Load test jsonl ground truth
# =========================

test_rows = []

with open(test_path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()

        if not line:
            continue

        obj = json.loads(line)
        prompt = extract_prompt(obj.get("prompt", ""))
        gt = extract_ground_truth(obj)

        test_rows.append(
            {
                "prompt": prompt,
                "_norm_prompt": norm_prompt(prompt),
                "groundtruth": to_str(gt),
            }
        )

test_df = pd.DataFrame(test_rows)

if len(test_df) == 0:
    raise ValueError(f"Empty test jsonl: {test_path}")


# =========================
# Merge ground truth
# =========================

gen_df = gen_df.copy()
gen_df["_row_id"] = range(len(gen_df))
gen_df["_norm_prompt"] = gen_df["prompt"].map(norm_prompt)

gt_map = {}
duplicate_prompts = set()

for row in test_df.to_dict("records"):
    key = row["_norm_prompt"]

    if key in gt_map:
        duplicate_prompts.add(key)

    gt_map[key] = row["groundtruth"]

if duplicate_prompts:
    print(
        f"[WARN] Found {len(duplicate_prompts)} duplicate normalized prompts in test data. "
        f"Prompt-based matching may use the last occurrence."
    )

gen_df["groundtruth"] = gen_df["_norm_prompt"].map(gt_map)

missing_mask = gen_df["groundtruth"].isna()
missing_count = int(missing_mask.sum())

if missing_count > 0:
    if len(gen_df) == len(test_df):
        print(
            f"[WARN] {missing_count}/{len(gen_df)} rows failed prompt-based ground truth matching. "
            f"Falling back to row-order matching because lengths match."
        )

        gen_df["groundtruth"] = test_df["groundtruth"].tolist()
    else:
        examples = gen_df.loc[missing_mask, "prompt"].head(3).tolist()

        raise ValueError(
            f"Failed to match ground truth for {missing_count}/{len(gen_df)} rows, "
            f"and cannot fall back by row order because generation rows={len(gen_df)}, "
            f"test rows={len(test_df)}. Examples: {examples}"
        )


# =========================
# Build enriched columns
# =========================

gen_df["response_texts"] = gen_df["responses"].map(extract_response_texts)
gen_df["answers"] = gen_df["responses"].map(extract_answers)

# Important:
# overwrite responses with normalized plain-text list.
# This makes downstream evaluators safer if they expect text responses.
gen_df["responses"] = gen_df["response_texts"]

# Compatibility aliases.
gen_df["ground_truth"] = gen_df["groundtruth"].map(to_str)
gen_df["task"] = task

empty_answer_count = int(
    gen_df["answers"].map(lambda xs: sum(1 for x in xs if x == "")).sum()
)

all_empty_rows = int(
    gen_df["answers"].map(lambda xs: all(x == "" for x in xs)).sum()
)

bad_len_rows = int(
    gen_df["answers"].map(lambda xs: len(xs) != k).sum()
)

if empty_answer_count > 0:
    print(
        f"[WARN] Found {empty_answer_count} empty extracted answers. "
        f"Rows with all-{k}-empty answers: {all_empty_rows}."
    )

if bad_len_rows > 0:
    raise ValueError(f"Internal error: {bad_len_rows} rows have answers length != k")


# =========================
# Write enriched parquet
# =========================

preferred_cols = [
    "prompt",
    "responses",
    "response_texts",
    "answers",
    "groundtruth",
    "ground_truth",
    "task",
    "_row_id",
]

other_cols = [
    c
    for c in gen_df.columns
    if c not in preferred_cols and c != "_norm_prompt"
]

out_cols = preferred_cols + other_cols

out_df = gen_df[out_cols].copy()
out_df["prompt"] = out_df["prompt"].map(lambda x: to_str(extract_prompt(x)))
out_df["groundtruth"] = out_df["groundtruth"].map(to_str)
out_df["ground_truth"] = out_df["ground_truth"].map(to_str)

out_path.parent.mkdir(parents=True, exist_ok=True)
out_df.to_parquet(out_path, index=False)

print(f"[INFO] Wrote enriched parquet: {out_path}")
print(f"[INFO] Rows: {len(out_df)}")
print(f"[INFO] Columns: {list(out_df.columns)}")
print(f"[INFO] Empty extracted answers: {empty_answer_count}")
print(f"[INFO] Rows with all answers empty: {all_empty_rows}")
print("[INFO] First groundtruth:", out_df["groundtruth"].iloc[0])
print("[INFO] First answers:", out_df["answers"].iloc[0])

try:
    first_responses = out_df["responses"].iloc[0]
    first_preview = first_responses[0][:300] if first_responses else ""
    print("[INFO] First response preview:", first_preview)
except Exception as exc:
    print("[WARN] Could not print first response preview:", repr(exc))
PY
}

evaluate_task() {
  local task="$1"

  local enriched_file="${ENRICHED_DIR}/${task}_${MODEL_NAME}_k${K}.with_gt.parquet"
  local output_json="${EVAL_DIR}/${task}_${MODEL_NAME}_k${K}.eval.json"
  local output_jsonl="${EVAL_DIR}/${task}_${MODEL_NAME}_k${K}.eval.details.jsonl"
  local output_jsonl_alias="${EVAL_DIR}/${task}_${MODEL_NAME}_k${K}.eval.jsonl"

  if [ ! -f "${enriched_file}" ]; then
    echo "[ERROR] Missing enriched parquet file: ${enriched_file}"
    exit 1
  fi

  echo "[INFO] Evaluating task=${task}"
  echo "[INFO] Eval input: ${enriched_file}"
  echo "[INFO] Output json: ${output_json}"
  echo "[INFO] Details jsonl: ${output_jsonl}"
  echo "[INFO] Details alias jsonl: ${output_jsonl_alias}"

  "${PYTHON_BIN}" - "${enriched_file}" "${K}" <<'PY'
import sys
import pandas as pd

p = sys.argv[1]
k = int(sys.argv[2])

df = pd.read_parquet(p)

print("[DEBUG] Eval input sanity check")
print("[DEBUG] path:", p)
print("[DEBUG] shape:", df.shape)
print("[DEBUG] columns:", list(df.columns))

if len(df) == 0:
    raise ValueError(f"Eval input parquet is empty: {p}")

required = ["prompt", "responses", "answers", "groundtruth"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Eval input missing required columns: {missing}")

bad_len = int(df["answers"].map(lambda xs: len(xs) != k).sum())
empty_answers = int(df["answers"].map(lambda xs: sum(1 for x in xs if x == "")).sum())
all_empty_rows = int(df["answers"].map(lambda xs: all(x == "" for x in xs)).sum())

print("[DEBUG] bad answer-list length rows:", bad_len)
print("[DEBUG] empty answers:", empty_answers)
print("[DEBUG] all-empty rows:", all_empty_rows)
print("[DEBUG] first groundtruth:", df["groundtruth"].iloc[0])
print("[DEBUG] first answers:", df["answers"].iloc[0])

if bad_len > 0:
    raise ValueError(f"{bad_len} rows have answers length != k")

if all_empty_rows == len(df):
    raise ValueError(
        "All rows have empty extracted answers. "
        "Check generation response format or answer extraction logic."
    )
PY

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

  if [ -f "${output_jsonl}" ]; then
    cp "${output_jsonl}" "${output_jsonl_alias}"
    echo "[INFO] Copied details jsonl alias: ${output_jsonl_alias}"

    line_count="$(wc -l < "${output_jsonl}" | tr -d ' ')"
    echo "[INFO] Details jsonl line count: ${line_count}"

    if [ "${line_count}" = "0" ]; then
      echo "[WARN] Details jsonl is empty: ${output_jsonl}"
      echo "[WARN] The enriched parquet is non-empty, so this likely means evaluation.py skipped all rows."
      echo "[WARN] Check recipe/GARO/scripts/evaluate/evaluation.py expected input schema."
    fi
  else
    echo "[WARN] evaluation.py did not create details jsonl: ${output_jsonl}"
  fi

  if [ ! -f "${output_json}" ]; then
    echo "[WARN] evaluation.py did not create output json: ${output_json}"
  fi

  echo "[INFO] Done evaluating: ${task}"
}

echo "[INFO] ==============================="
echo "[INFO] Stage 1/2: merge ground truth and extracted answers"
echo "[INFO] ==============================="

for task in "${TASKS[@]}"; do
  merge_gt_for_task "${task}"
done

echo "[INFO] All merged parquet files prepared."

echo "[INFO] ==============================="
echo "[INFO] Stage 2/2: evaluate"
echo "[INFO] ==============================="

for task in "${TASKS[@]}"; do
  evaluate_task "${task}"
done

echo "[INFO] All evaluations completed."