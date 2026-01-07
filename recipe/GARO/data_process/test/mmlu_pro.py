#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


INSTRUCTION_SUFFIX = (
    "\n\nPlease choose the single best answer. "
    "Reason step by step and output your final answer as only one letter "
    "(A, B, C, D, E, F, G, H, I, or J) inside \\boxed{}."
)


def _to_str(x):
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _extract_question_and_options(ex: dict):
    """Best-effort extraction across different MMLU-Pro dataset schemas."""
    q = _to_str(ex.get("question") or ex.get("prompt") or ex.get("input") or "").strip()

    # Options can be: list[str] (preferred), dict, or nested under various keys.
    opts = ex.get("options")
    if opts is None:
        opts = ex.get("choices")
    if opts is None:
        opts = ex.get("answer_choices")

    options = []
    if isinstance(opts, list):
        options = [_to_str(o).strip() for o in opts]
    elif isinstance(opts, dict):
        # dict like {"A": "...", ...} or {"choices": [...]} etc.
        if all(k in opts for k in list("ABCDEFGHIJ")):
            options = [_to_str(opts[k]).strip() for k in list("ABCDEFGHIJ")]
        elif "choices" in opts and isinstance(opts["choices"], list):
            options = [_to_str(o).strip() for o in opts["choices"]]

    # Trim to at most 10 options for safety.
    options = [o for o in options if o]
    if len(options) > 10:
        options = options[:10]

    return q, options


def _format_options(options):
    letters = list("ABCDEFGHIJ")
    lines = []
    for i, opt in enumerate(options[:10]):
        lines.append(f"({letters[i]}) {opt}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="保存 JSONL/Parquet 的目录（必填）",
    )
    return parser.parse_args()


def to_letter(ans):
    """
    mmlu-pro 的 answer 可能是：
    - int: 0..9
    - str: 'A'..'J' 或 '0'..'9'
    统一转成 'A'~'J'
    """
    if isinstance(ans, int):
        mapping = list("ABCDEFGHIJ")
        return mapping[ans] if 0 <= ans < 10 else str(ans)

    s = str(ans).strip()
    if s in list("ABCDEFGHIJ"):
        return s
    if s.isdigit():
        idx = int(s)
        mapping = list("ABCDEFGHIJ")
        if 0 <= idx < 10:
            return mapping[idx]
    return s


def make_schema():
    prompt_struct = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ])
    return pa.schema([
        ("prompt", pa.list_(prompt_struct)),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
    ])


def main():
    args = parse_args()
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    output_jsonl = os.path.join(save_dir, "mmlu_pro.jsonl")
    output_parquet = os.path.join(save_dir, "mmlu_pro.parquet")

    # MMLU-Pro (10 options)
    # Prefer TIGER-Lab/MMLU-Pro; keep a fallback for environments without it.
    try:
        print("Loading TIGER-Lab/MMLU-Pro (split='test')...")
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception as e:
        print(f"[WARN] Failed to load TIGER-Lab/MMLU-Pro: {e}")
        print("Falling back to cais/mmlu (config='all', split='test')... (NOTE: this is NOT MMLU-Pro)")
        ds = load_dataset("cais/mmlu", "all", split="test")

    records = []
    for ex in ds:
        q, options = _extract_question_and_options(ex)
        ans = ex.get("answer", ex.get("label", ""))
        gt = to_letter(ans)

        opt_block = _format_options(options)
        prompt_text = (
            q
            + ("\n\nOptions:\n" + opt_block if opt_block else "")
            + INSTRUCTION_SUFFIX
        )
        prompt_messages = [
            {"role": "user", "content": prompt_text}
        ]

        records.append(
            {
                "prompt": prompt_messages,
                "reward_model": {"ground_truth": gt},
                "data_source": "mmlu_pro",
            }
        )

    print(f"Total samples: {len(records)}")

    # 写 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写 Parquet（显式 schema，保证 prompt 类型是 list<struct<role, content>>）
    schema = make_schema()
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, output_parquet)

    print(f"Saved to:\n  {output_jsonl}\n  {output_parquet}")


if __name__ == "__main__":
    main()
