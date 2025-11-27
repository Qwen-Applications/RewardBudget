#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process nvidia/OpenMathReasoning into VERL-friendly train/valid datasets via streaming.

HF fields (for cot/tir/genselect splits):
  expected_answer: str
  problem: str
  generated_solution: str
  pass_rate_72b_tir: str (float string or "n/a")

Output fields:
  prompt: "Solve the following problem. Make sure to put the answer (and only answer) inside \boxed{}. " + problem
  response: generated_solution
  reward_model: {"ground_truth": expected_answer}
  data_source: "OpenMath"
  ability: "math"
  pass_rate_72b_tir: float

Filtering:
  - token length: prompt <= max_prompt_length, response <= max_response_length
  - min_pass_rate < pass_rate_72b_tir < max_pass_rate
  - keep first max_train_data valid to train, next max_valid_data to valid

Saving:
  - jsonl + parquet to train_data_dir / valid_data_dir
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PROMPT_PREFIX = (
    "Solve the following problem. Make sure to put the answer "
    "(and only answer) inside \boxed{}. "
)


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
    p.add_argument("--min_pass_rate", type=float, default=0.5,
                   help="Filter out samples with pass_rate_72b_tir <= this value.")
    p.add_argument("--max_pass_rate", type=float, default=1.0,
                   help="Filter out samples with pass_rate_72b_tir >= this value.")
    p.add_argument("--hf_splits", type=str, default="cot",
                   help="Comma-separated HF splits to stream in order.")
    p.add_argument("--chunk_size", type=int, default=1024,
                   help="Flush parquet every N samples to limit RAM.")
    return p.parse_args()


def make_schema():
    return pa.schema([
        ("prompt", pa.string()),
        ("response", pa.string()),
        ("reward_model", pa.struct([("ground_truth", pa.string())])),
        ("data_source", pa.string()),
        ("ability", pa.string()),
        ("pass_rate_72b_tir", pa.float32()),
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

    splits = [s.strip() for s in args.hf_splits.split(",") if s.strip()]

    train_jsonl = train_dir / "OpenMathReasoning_train.jsonl"
    valid_jsonl = valid_dir / "OpenMathReasoning_valid.jsonl"
    train_parquet = train_dir / "OpenMathReasoning_train.parquet"
    valid_parquet = valid_dir / "OpenMathReasoning_valid.parquet"

    schema = make_schema()
    train_writer = pq.ParquetWriter(train_parquet, schema, compression="zstd")
    valid_writer = pq.ParquetWriter(valid_parquet, schema, compression="zstd")

    train_buf, valid_buf = [], []
    train_kept, valid_kept, seen = 0, 0, 0

    with train_jsonl.open("w", encoding="utf-8") as f_tr, \
         valid_jsonl.open("w", encoding="utf-8") as f_va:

        for split in splits:
            ds = load_dataset(
                "nvidia/OpenMathReasoning",
                split=split,
                streaming=True,
            )

            pbar = tqdm(ds, desc=f"Streaming {split}", unit="ex")
            for ex in pbar:
                seen += 1

                problem = ex.get("problem")
                response = ex.get("generated_solution")
                gt = ex.get("expected_answer")
                pr_raw = ex.get("pass_rate_72b_tir")

                # skip if required fields missing (e.g., additional_problems)
                if problem is None or response is None or gt is None or pr_raw is None:
                    continue

                # filter to keep only purely numeric expected_answer
                gt_stripped = gt.strip()
                if not gt_stripped.isdigit():
                    continue

                # parse pass_rate_72b_tir (string -> float), skip non-numeric like "n/a"
                try:
                    pr = float(pr_raw)
                except Exception:
                    continue

                if pr >= args.max_pass_rate or pr <= args.min_pass_rate:
                    continue

                prompt = PROMPT_PREFIX + problem

                # token length filtering
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                resp_len = len(tokenizer.encode(response, add_special_tokens=False))
                if prompt_len > args.max_prompt_length or resp_len > args.max_response_length:
                    continue

                row = {
                    "prompt": prompt,
                    "response": response,
                    "reward_model": {"ground_truth": gt_stripped},
                    "data_source": "OpenMath",
                    "ability": "math",
                    "pass_rate_72b_tir": pr,
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
                    "valid_kept": valid_kept,
                    "split": split
                })

            if train_kept >= args.max_train_data and valid_kept >= args.max_valid_data:
                break

    # flush tail and close
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
