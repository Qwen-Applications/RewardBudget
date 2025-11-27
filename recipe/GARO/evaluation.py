# recipe/GARO/eval_pass_k.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Optional

from recipe.GARO.reward import compute_score


def eval_pass_at_k(file_path: str, k: int, output_path: Optional[str] = None) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    total = 0
    pass1_correct = 0.0
    passk_correct = 0.0  # pass@k: 看前 k 个 responses 是否有至少一个答对

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # 已知你的数据结构是固定的，不做健壮处理
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

    metrics = {
        "file": file_path,
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
