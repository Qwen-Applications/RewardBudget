# recipe/GARO/eval_pass_k.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Optional

import torch
import pyarrow.parquet as pq

from recipe.GARO.reward import compute_score

def is_main_process() -> bool:
    """只在未初始化 DDP 或 rank0 时返回 True。"""
    return (not torch.distributed.is_initialized() 
            or torch.distributed.get_rank() == 0)


def local_mkdir(path: str) -> str:
    """只在 main process 创建目录，然后所有 rank 同步一下。"""
    if path == "":
        return path

    if is_main_process():
        os.makedirs(path, exist_ok=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return path

def _iter_records(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    # --- JSONL 模式 ---
    if ext in [".jsonl", ".json"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    # --- Parquet 模式 ---
    elif ext == ".parquet":
        table = pq.read_table(file_path)
        cols = table.to_pydict()         # dict: col_name -> list
        n = table.num_rows

        for i in range(n):
            row = {k: v[i] for k, v in cols.items()}
            yield row
    else:
        raise ValueError(f"Unsupported file extension for {file_path}")

def eval_pass_at_k(file_path: str, k: int, output_path: Optional[str] = None) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    total = 0
    pass1_correct = 0.0
    passk_correct = 0.0  # pass@k: 看前 k 个 responses 是否有至少一个答对

    for obj in _iter_records(file_path):
        responses = obj["responses"]                       # list[str]
        gt = obj["reward_model"]["ground_truth"]           # str
        data_source: str = obj.get("data_source", "unknown")

        if not responses:
            continue

        total += 1

        # ----- pass@1: 只看 responses[0] -----
        pass1_correct += compute_score(
            data_source=data_source,
            solution_str=responses[0],
            ground_truth=gt,
        )

        # ----- pass@k: 只看前 k 个 responses，有一个对就算 1 -----
        num_to_use = min(k, len(responses))
        any_correct_k = 0.0
        for sol in responses[:num_to_use]:
            score = compute_score(
                data_source=data_source,
                solution_str=sol,
                ground_truth=gt,
            )
            if score > 0:
                any_correct_k = 1.0
                break

        passk_correct += any_correct_k

    if total == 0:
        print("No valid samples found in file.")
        return

    pass1_acc = pass1_correct / total
    passk_acc = passk_correct / total

    print(f"File: {file_path}")
    print(f"Total samples: {total}")
    print(f"Pass@1 accuracy: {pass1_acc:.4%}")
    print(f"Pass@{k} accuracy (any of first {k} responses correct): {passk_acc:.4%}")

    # ===== 把结果保存到文件 =====
    if output_path is None:
        output_path = "recipe/GARO/evaluation_results.json"

    base_name = os.path.basename(file_path)          # e.g. "drgrpo_math500.parquet"
    file_stem, _ = os.path.splitext(base_name)      # e.g. "drgrpo_math500"

    metrics = {
        "file": file_path,
        "file_name": file_stem,
        "total_samples": total,
        "k": k,
        "pass_at_1": {
            "correct": pass1_correct,
            "accuracy": pass1_acc,
        },
        "pass_at_k": {
            "correct": passk_correct,
            "accuracy": passk_acc,
        },
    }

    if is_main_process():
        # 确保上级目录存在（在 main process 上建）
        dir_path = os.path.dirname(output_path)
        local_mkdir(dir_path)

        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(metrics, fout, ensure_ascii=False, indent=2)

        print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to JSONL file with fields: prompt, responses, reward_model.ground_truth",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="k for pass@k. We look at the first k responses for each sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON (default: <file>.pass_at_k.json)",
    )
    args = parser.parse_args()

    eval_pass_at_k(args.file, args.k, args.output)
