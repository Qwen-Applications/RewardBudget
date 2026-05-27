#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


HF_NAME = "HuggingFaceH4/aime_2024"
DATA_SOURCE = "aime24"
SPLIT = "train"
PROMPT_KEY = "problem"
GROUNDTRUTH_KEY = "answer"
ABILITY = "math"
PROMPT_PREFIX = "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}. "


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where the processed JSONL and Parquet files are saved.",
    )
    return parser.parse_args()


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def make_schema() -> pa.Schema:
    prompt_struct = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ])
    return pa.schema([
        ("prompt", pa.list_(prompt_struct)),
        ("response", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
        ("ability", pa.string()),
    ])


def build_records() -> List[Dict[str, Any]]:
    ds = load_dataset(HF_NAME, split=SPLIT)
    records: List[Dict[str, Any]] = []
    for ex in ds:
        question = _to_str(ex.get(PROMPT_KEY, "")).strip()
        ground_truth = _to_str(ex.get(GROUNDTRUTH_KEY, "")).strip()
        if not question:
            continue
        records.append(
            {
                "prompt": [{"role": "user", "content": (PROMPT_PREFIX + question).strip()}],
                "response": "",
                "reward_model": {"ground_truth": ground_truth},
                "data_source": DATA_SOURCE,
                "ability": ABILITY,
            }
        )
    return records


def write_records(save_dir: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"{DATA_SOURCE}.jsonl")
    parquet_path = os.path.join(save_dir, f"{DATA_SOURCE}.parquet")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    table = pa.Table.from_pylist(records, schema=make_schema())
    pq.write_table(table, parquet_path)
    print(f"Saved {len(records)} rows -> {jsonl_path} / {parquet_path}")


def main() -> None:
    args = parse_args()
    write_records(args.save_dir, build_records())


if __name__ == "__main__":
    main()
