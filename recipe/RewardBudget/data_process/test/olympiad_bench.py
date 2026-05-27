#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


PROMPT_PREFIX = "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}. "
DATA_SOURCE = "OlympiadBench_physics_en_comp"
ABILITY = "physics"
TASK_TO_CONFIGS = {
    DATA_SOURCE: ["OE_TO_physics_en_COMP", "TP_TO_physics_en_COMP"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where the processed OlympiadBench files are saved.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace split to load. Defaults to train.",
    )
    return parser.parse_args()


def _to_str(value: Any) -> str:
    """Convert lists, dictionaries, and other values into strings that can be written to disk."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
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


def _write_records(save_dir: str, task_name: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    jsonl_path = os.path.join(save_dir, f"{task_name}.jsonl")
    parquet_path = os.path.join(save_dir, f"{task_name}.parquet")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    table = pa.Table.from_pylist(records, schema=make_schema())
    pq.write_table(table, parquet_path)
    print(f"Saved {len(records)} rows -> {jsonl_path} / {parquet_path}")


def main() -> None:
    args = parse_args()

    for task_name, configs in TASK_TO_CONFIGS.items():
        records: List[Dict[str, Any]] = []
        for config in configs:
            print(f"Loading subset: {config} (split='{args.split}')")
            ds = load_dataset("Hothan/OlympiadBench", config, split=args.split)

            for ex in ds:
                question = _to_str(ex.get("question", "")).strip()
                final_answer = _to_str(ex.get("final_answer", "")).strip()
                if not question:
                    continue

                records.append(
                    {
                        "prompt": [{"role": "user", "content": (PROMPT_PREFIX + question).strip()}],
                        "response": "",
                        "reward_model": {"ground_truth": final_answer},
                        "data_source": task_name,
                        "ability": ABILITY,
                    }
                )

        _write_records(args.save_dir, task_name, records)

    print("Done.")


if __name__ == "__main__":
    main()
