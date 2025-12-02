#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
from typing import Optional, List, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_NAME = "zhuzilin/dapo-math-17k"
BASENAME = "dapo_math_17k"


def parse_args():
    p = argparse.ArgumentParser()

    # ====== 输出路径相关 ======
    p.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Dir to save train jsonl/parquet.",
    )
    p.add_argument(
        "--valid_data_dir",
        type=str,
        required=True,
        help="Dir to save valid jsonl/parquet.",
    )

    # ====== tokenizer & 长度控制 ======
    p.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="HF tokenizer id/path.",
    )
    p.add_argument(
        "--max_prompt_length",
        type=int,
        default=1024,
        help="Max prompt length in tokens.",
    )
    p.add_argument(
        "--max_response_length",
        type=int,
        default=8192,
        help="Max response length in tokens.",
    )

    # ====== chunk & split 相关 ======
    p.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Flush parquet every N samples to limit RAM.",
    )
    p.add_argument(
        "--valid_size",
        type=int,
        default=1000,
        help="Target number of validation examples (used when valid_ratio <= 0).",
    )
    p.add_argument(
        "--valid_ratio",
        type=float,
        default=0.0,
        help="Portion of data used as validation (approximate). If <= 0, use valid_size instead.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid splitting.",
    )

    return p.parse_args()


def is_int(label: str):
    try:
        int(label)
        return True
    except ValueError:
        return False


def build_record(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    将原始 example 映射为目标结构。
    原始字段：prompt: str, label: str
    """
    return {
        "prompt": example["prompt"],
        "response": "",
        "reward_model": {"ground_truth": example["label"]},
        "ability": "math",
        "data_source": "dapo-math-17k",
    }


def flush_buffer_to_parquet(
    buffer: List[Dict[str, Any]],
    writer: Optional[pq.ParquetWriter],
    parquet_path: str,
) -> pq.ParquetWriter:
    """
    将当前 buffer 写入 parquet，并清空 buffer。
    使用同一个 ParquetWriter，多次写入不同 row group。
    """
    if not buffer:
        return writer

    table = pa.Table.from_pylist(buffer)

    if writer is None:
        # 第一次写入时创建 writer，并锁定 schema
        writer = pq.ParquetWriter(parquet_path, table.schema)

    writer.write_table(table)
    buffer.clear()
    return writer


def main():
    args = parse_args()

    os.makedirs(args.train_data_dir, exist_ok=True)
    os.makedirs(args.valid_data_dir, exist_ok=True)

    train_jsonl_path = os.path.join(args.train_data_dir, f"{BASENAME}_train.jsonl")
    valid_jsonl_path = os.path.join(args.valid_data_dir, f"{BASENAME}_valid.jsonl")
    train_parquet_path = os.path.join(args.train_data_dir, f"{BASENAME}_train.parquet")
    valid_parquet_path = os.path.join(args.valid_data_dir, f"{BASENAME}_valid.parquet")

    print(f"[INFO] Train jsonl:   {train_jsonl_path}")
    print(f"[INFO] Valid jsonl:   {valid_jsonl_path}")
    print(f"[INFO] Train parquet: {train_parquet_path}")
    print(f"[INFO] Valid parquet: {valid_parquet_path}")

    # ====== tokenizer ======
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ====== streaming 读取数据集，避免一次性载入内存 ======
    ds_iter = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    rng = random.Random(args.seed)

    # ====== 打开 jsonl 文件 ======
    train_f = open(train_jsonl_path, "w", encoding="utf-8")
    valid_f = open(valid_jsonl_path, "w", encoding="utf-8")

    # parquet 写入相关
    train_buffer: List[Dict[str, Any]] = []
    valid_buffer: List[Dict[str, Any]] = []
    train_writer: Optional[pq.ParquetWriter] = None
    valid_writer: Optional[pq.ParquetWriter] = None

    # 统计信息
    total_seen = 0
    total_kept = 0
    total_train = 0
    total_valid = 0
    skipped_prompt = 0
    skipped_long_response = 0

    try:
        for example in ds_iter:
            total_seen += 1

            # ------- 过滤 -------
            prompt_text: str = example["prompt"][0]["content"]
            label_text: str = example["label"]

            # 计算 prompt token 长度
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            if len(prompt_ids) > args.max_prompt_length:
                skipped_prompt += 1
                continue

            # 过滤 label 非整数的数据
            if not is_int(label_text):
                skipped_prompt += 1
                continue

            response_text = ""
            if response_text:
                response_ids = tokenizer(
                    response_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
                if len(response_ids) > args.max_response_length:
                    skipped_long_response += 1
                    continue

            record = build_record({"prompt": prompt_text, "label": label_text})
            total_kept += 1

            # ------- train / valid 划分逻辑 -------
            # 默认用固定 valid_size，若 valid_ratio > 0 则退回比例模式
            if args.valid_ratio > 0.0:
                is_valid = rng.random() < args.valid_ratio
            else:
                # 优先填满 valid_size，之后全部进 train
                is_valid = total_valid < args.valid_size

            if is_valid:
                # 写 jsonl
                valid_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                # 写 parquet buffer
                valid_buffer.append(record)
                total_valid += 1

                if len(valid_buffer) >= args.chunk_size:
                    valid_writer = flush_buffer_to_parquet(
                        valid_buffer, valid_writer, valid_parquet_path
                    )
            else:
                train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                train_buffer.append(record)
                total_train += 1

                if len(train_buffer) >= args.chunk_size:
                    train_writer = flush_buffer_to_parquet(
                        train_buffer, train_writer, train_parquet_path
                    )

    finally:
        # ====== 收尾：flush 剩余 buffer & 关闭 writer / 文件 ======
        train_writer = flush_buffer_to_parquet(train_buffer, train_writer, train_parquet_path)
        valid_writer = flush_buffer_to_parquet(valid_buffer, valid_writer, valid_parquet_path)

        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()

        train_f.close()
        valid_f.close()

    # ====== 打印总结 ======
    print("\n[STATS]")
    print(f"  Total examples seen:         {total_seen}")
    print(f"  Kept after length filter:    {total_kept}")
    print(f"    -> Train:                  {total_train}")
    print(f"    -> Valid:                  {total_valid}")


if __name__ == "__main__":
    main()
