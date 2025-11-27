#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process nvidia/OpenScienceReasoning-2 into VERL-friendly format with streaming,
then split into train/valid locally (dataset has only train split on HF).

Input keys:
  expected_answer: str
  input: str
  output: str

Output keys:
  prompt: data["input"]
  response: data["output"]
  reward_model: {"ground_truth": data["expected_answer"]}
  data_source: "OpenScience"
  ability: "knowledge"

Features:
  - streaming HF dataset to avoid full download
  - filter by max_prompt_length / max_response_length (token count)
  - keep first max_train_data valid samples as train
  - keep next max_valid_data valid samples as valid
  - save jsonl and parquet into train_data_dir / valid_data_dir
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data_dir", type=str, required=True,
                   help="Dir to save train jsonl/parquet.")
    p.add_argument("--valid_data_dir", type=str, required=True,
                   help="Dir to save valid jsonl/parquet.")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="HF tokenizer id/path.")
    p.add_argument("--max_prompt_length", type=int, default=1024,
                   help="Max prompt length in tokens.")
    p.add_argument("--max_response_length", type=int, default=8192,
                   help="Max response length in tokens.")
    p.add_argument("--max_train_data", type=int, default=10000,
                   help="Number of valid train samples to keep.")
    p.add_argument("--max_valid_data", type=int, default=1000,
                   help="Number of valid valid samples to keep.")
    p.add_argument("--chunk_size", type=int, default=10000,
                   help="Flush parquet every N samples to limit RAM.")
    return p.parse_args()


def make_schema():
    return pa.schema([
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
        ("ability", pa.string()),
    ])


def flush_buf(buf, writer, schema):
    if not buf:
        return
    table = pa.Table.from_pylist(buf, schema=schema)
    writer.write_table(table)
    buf.clear()


def main():
    args = parse_args()

    train_dir = Path(args.train_data_dir)
    valid_dir = Path(args.valid_data_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, use_fast=True
    )

    # streaming read (won't download full dataset)
    ds = load_dataset(
        "nvidia/OpenScienceReasoning-2",
        split="train",
        streaming=True,
    )

    train_jsonl = train_dir / "OpenScienceReasoning2_train.jsonl"
    valid_jsonl = valid_dir / "OpenScienceReasoning2_valid.jsonl"
    train_parquet = train_dir / "OpenScienceReasoning2_train.parquet"
    valid_parquet = valid_dir / "OpenScienceReasoning2_valid.parquet"

    schema = make_schema()
    train_writer = pq.ParquetWriter(train_parquet, schema, compression="zstd")
    valid_writer = pq.ParquetWriter(valid_parquet, schema, compression="zstd")

    train_buf, valid_buf = [], []
    train_kept, valid_kept, seen = 0, 0, 0

    with train_jsonl.open("w", encoding="utf-8") as f_tr, \
         valid_jsonl.open("w", encoding="utf-8") as f_va:

        pbar = tqdm(ds, desc="Streaming + filtering + split", unit="ex")
        for ex in pbar:
            seen += 1
            prompt = ex.get("input")
            response = ex.get("output")
            gt = ex.get("expected_answer")
            if prompt is None or response is None or gt is None:
                continue

            # token length filtering
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            resp_len = len(tokenizer.encode(response, add_special_tokens=False))
            if prompt_len > args.max_prompt_length or resp_len > args.max_response_length:
                continue

            row = {
                "prompt": prompt,
                "response": response,
                "reward_model": {"ground_truth": gt},
                "data_source": "OpenScience",
                "ability": "knowledge",
            }

            if train_kept < args.max_train_data:
                f_tr.write(json.dumps(row, ensure_ascii=False) + "\n")
                train_buf.append(row)
                train_kept += 1
                if len(train_buf) >= args.chunk_size:
                    flush_buf(train_buf, train_writer, schema)

            elif valid_kept < args.max_valid_data:
                f_va.write(json.dumps(row, ensure_ascii=False) + "\n")
                valid_buf.append(row)
                valid_kept += 1
                if len(valid_buf) >= args.chunk_size:
                    flush_buf(valid_buf, valid_writer, schema)

            else:
                break

            pbar.set_postfix({
                "seen": seen,
                "train_kept": train_kept,
                "valid_kept": valid_kept
            })

    # flush tail
    flush_buf(train_buf, train_writer, schema)
    flush_buf(valid_buf, valid_writer, schema)
    train_writer.close()
    valid_writer.close()

    print(f"Done. Seen={seen}, train_kept={train_kept}, valid_kept={valid_kept}")
    print(f"Train JSONL: {train_jsonl}")
    print(f"Train Parquet: {train_parquet}")
    print(f"Valid JSONL: {valid_jsonl}")
    print(f"Valid Parquet: {valid_parquet}")


if __name__ == "__main__":
    main()
