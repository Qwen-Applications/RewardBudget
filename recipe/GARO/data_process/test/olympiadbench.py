#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


PROMPT_PREFIX = (
    r"Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}. "
)


# The evaluation scripts expect these 8 parquet/jsonl files under test_data/:
#   OlympiadBench_{math|physics}_{en|zh}_{comp|cee}
# and each task merges OE + TP subsets.
TASK_TO_CONFIGS = {
    "OlympiadBench_math_en_comp": ["OE_TO_maths_en_COMP", "TP_TO_maths_en_COMP"],
    "OlympiadBench_math_en_cee": ["OE_TO_maths_en_CEE", "TP_TO_maths_en_CEE"],
    "OlympiadBench_math_zh_comp": ["OE_TO_maths_zh_COMP", "TP_TO_maths_zh_COMP"],
    "OlympiadBench_math_zh_cee": ["OE_TO_maths_zh_CEE", "TP_TO_maths_zh_CEE"],
    "OlympiadBench_physics_en_comp": ["OE_TO_physics_en_COMP", "TP_TO_physics_en_COMP"],
    "OlympiadBench_physics_en_cee": ["OE_TO_physics_en_CEE", "TP_TO_physics_en_CEE"],
    "OlympiadBench_physics_zh_comp": ["OE_TO_physics_zh_COMP", "TP_TO_physics_zh_COMP"],
    "OlympiadBench_physics_zh_cee": ["OE_TO_physics_zh_CEE", "TP_TO_physics_zh_CEE"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="保存 OlympiadBench 处理结果的目录（必填）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split（默认 train）",
    )
    return parser.parse_args()


def _to_str(x):
    """把 list/dict 等统一转成可落盘的字符串。"""
    if x is None:
        return ""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


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


def _write_records(save_dir: str, task_name: str, records: list):
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"{task_name}.jsonl")
    parquet_path = os.path.join(save_dir, f"{task_name}.parquet")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    schema = make_schema()
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, parquet_path)

    print(f"Saved {len(records)} rows -> {jsonl_path} / {parquet_path}")


def main():
    args = parse_args()
    save_dir = args.save_dir
    split = args.split

    for task_name, cfgs in TASK_TO_CONFIGS.items():
        records = []
        for cfg in cfgs:
            print(f"Loading subset: {cfg} (split='{split}')")
            ds = load_dataset("Hothan/OlympiadBench", cfg, split=split)

            for ex in ds:
                question = _to_str(ex.get("question", "")).strip()
                final_answer = _to_str(ex.get("final_answer", "")).strip()
                if not question:
                    continue

                prompt_messages = [
                    {"role": "user", "content": (PROMPT_PREFIX + question).strip()}
                ]
                records.append(
                    {
                        "prompt": prompt_messages,
                        "reward_model": {"ground_truth": final_answer},
                        "data_source": task_name,
                    }
                )

        _write_records(save_dir, task_name, records)

    print("Done.")


if __name__ == "__main__":
    main()
