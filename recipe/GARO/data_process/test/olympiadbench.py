#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

# 只处理 text-only 的 subset（名字里带 TO）
TEXT_ONLY_CONFIGS = [
    "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE",
    "OE_TO_maths_zh_COMP",
    "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE",
    "TP_TO_maths_en_COMP",
    "TP_TO_maths_zh_CEE",
    "TP_TO_maths_zh_COMP",
    "TP_TO_physics_en_COMP",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="保存 OlympiadBench 处理结果的目录（必填）",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="OlympiadBench_textonly_all",
        help="输出文件名不含后缀（默认合并后的大集合名）",
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
        ("response", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
    ])


def main():
    args = parse_args()
    save_dir = args.save_dir
    data_name = args.data_name

    os.makedirs(save_dir, exist_ok=True)

    output_jsonl = os.path.join(save_dir, f"{data_name}.jsonl")
    output_parquet = os.path.join(save_dir, f"{data_name}.parquet")

    records = []

    for cfg in TEXT_ONLY_CONFIGS:
        print(f"Loading subset: {cfg}")
        ds = load_dataset("Hothan/OlympiadBench", cfg, split="train")

        for ex in ds:
            question = _to_str(ex.get("question", "")).strip()
            solution = ex.get("solution", "")
            final_answer = ex.get("final_answer", "")

            if not question:
                continue

            prompt_messages = [
                {"role": "user", "content": question}
            ]

            record = {
                "prompt": prompt_messages,
                "response": _to_str(solution),
                "reward_model": {"ground_truth": _to_str(final_answer)},
                "data_source": "olympiadbench",
            }
            records.append(record)

    print(f"Total merged text-only samples: {len(records)}")

    # 写 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写 Parquet（显式 schema）
    schema = make_schema()
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, output_parquet)

    print(f"Saved to:\n  {output_jsonl}\n  {output_parquet}")
    print("Done.")


if __name__ == "__main__":
    main()
