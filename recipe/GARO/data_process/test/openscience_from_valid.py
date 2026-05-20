#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the OpenScience valid split produced by
recipe/GARO/data_process/train_valid/openscience.py into GARO test-data format.

Input valid format from train_valid/openscience.py:
  prompt: list[{role, content}]
  response: str
  reward_model: {ground_truth: str}
  data_source: "OpenScience"
  ability: "knowledge"

Output test format used by recipe/GARO/data_process/test/*.py:
  prompt: list[{role, content}]
  reward_model: {ground_truth: str}
  data_source: "openscience"  # aligned with test_data/openscience.parquet

Default output:
  /primus_datasets/zmy/GARO/test_data/openscience.parquet
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = "/primus_datasets/zmy/GARO/train_data/OpenScienceReasoning2_valid.parquet"
DEFAULT_OUTPUT = "/primus_datasets/zmy/GARO/test_data/openscience.parquet"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to OpenScienceReasoning2_valid.parquet produced by train_valid/openscience.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to write the converted test parquet.",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="openscience",
        help="data_source value to store in the test parquet.",
    )
    parser.add_argument(
        "--write_jsonl",
        action="store_true",
        help="Also write a sibling .jsonl file for inspection/debugging.",
    )
    return parser.parse_args()


def make_schema() -> pa.Schema:
    prompt_struct = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ])
    return pa.schema([
        ("prompt", pa.list_(prompt_struct)),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
    ])


def _is_chat_messages(x: Any) -> bool:
    if not isinstance(x, list) or len(x) == 0:
        return False
    return all(isinstance(m, dict) and "role" in m and "content" in m for m in x)


def normalize_prompt(prompt: Any) -> List[Dict[str, str]]:
    """Ensure prompt is HF chat-template messages: [{role, content}, ...]."""
    if _is_chat_messages(prompt):
        return [
            {
                "role": str(m.get("role", "user")),
                "content": "" if m.get("content") is None else str(m.get("content")),
            }
            for m in prompt
        ]

    if isinstance(prompt, (dict, list)):
        prompt_text = json.dumps(prompt, ensure_ascii=False)
    elif prompt is None:
        prompt_text = ""
    else:
        prompt_text = str(prompt)

    return [{"role": "user", "content": prompt_text.strip()}]


def extract_ground_truth(row: Dict[str, Any]) -> str:
    """Extract reward_model.ground_truth, with a small fallback for raw rows."""
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth", "")
    else:
        try:
            gt = reward_model["ground_truth"]
        except Exception:
            gt = row.get("expected_answer", "")

    return "" if gt is None else str(gt)


def convert_one_file(input_path: Path, output_path: Path, data_source: str, write_jsonl: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {input_path}")
    table = pq.read_table(input_path)
    rows = table.to_pylist()

    required_columns = {"prompt", "reward_model"}
    missing = required_columns - set(table.column_names)
    if missing:
        raise ValueError(f"{input_path} missing required columns: {sorted(missing)}")

    records = []
    skipped = 0
    for row in rows:
        prompt_messages = normalize_prompt(row.get("prompt"))
        gt = extract_ground_truth(row)

        # Test/eval data should have usable prompt and ground truth.
        if not prompt_messages or not prompt_messages[0].get("content") or not gt:
            skipped += 1
            continue

        records.append({
            "prompt": prompt_messages,
            "reward_model": {"ground_truth": gt},
            "data_source": data_source,
        })

    schema = make_schema()
    out_table = pa.Table.from_pylist(records, schema=schema)

    # Atomic write: prevents leaving a half-written openscience.parquet if interrupted.
    with tempfile.NamedTemporaryFile(
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        pq.write_table(out_table, tmp_path, compression="zstd")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Written parquet: {output_path}")
    print(f"Rows written: {len(records)}")
    if skipped:
        print(f"Rows skipped because prompt/ground_truth was empty: {skipped}")

    if write_jsonl:
        jsonl_path = output_path.with_suffix(".jsonl")
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(jsonl_path.parent),
            prefix=f".{jsonl_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_jsonl = Path(f.name)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            os.replace(tmp_jsonl, jsonl_path)
        finally:
            if tmp_jsonl.exists():
                tmp_jsonl.unlink()
        print(f"Written jsonl:   {jsonl_path}")

    print("Output schema:")
    print(pq.ParquetFile(output_path).schema)


def main():
    args = parse_args()
    convert_one_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        data_source=args.data_source,
        write_jsonl=args.write_jsonl,
    )


if __name__ == "__main__":
    main()
