#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="保存 JSONL/Parquet 的目录",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="hle",
        help="输出文件前缀名（会生成 data_name.jsonl / data_name.parquet）",
    )
    return parser.parse_args()


def make_schema():
    prompt_struct = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ])
    return pa.schema([
        ("id", pa.string()),
        ("prompt", pa.list_(prompt_struct)),
        ("data_source", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("answer_type", pa.string()),
    ])


def main():
    args = parse_args()

    save_dir = args.save_dir
    data_name = args.data_name

    os.makedirs(save_dir, exist_ok=True)

    output_jsonl = os.path.join(save_dir, f"{data_name}.jsonl")
    output_parquet = os.path.join(save_dir, f"{data_name}.parquet")

    print("Loading cais/hle (split='test')...")
    ds = load_dataset("cais/hle", split="test")

    records = []
    for ex in ds:
        # 只保留 image 为 None 的样本
        if ex.get("image") is not None:
            continue

        ex_id = str(ex.get("id", ""))
        question = str(ex.get("question", ""))
        answer = str(ex.get("answer", ""))
        answer_type = str(ex.get("answer_type", ""))

        prompt_messages = [
            {"role": "user", "content": question}
        ]

        records.append(
            {
                "id": ex_id,
                "prompt": prompt_messages,
                "data_source": data_name,
                "reward_model": {"ground_truth": answer},
                "answer_type": answer_type,
            }
        )

    print(f"Kept {len(records)} text-only samples (image is None).")

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
