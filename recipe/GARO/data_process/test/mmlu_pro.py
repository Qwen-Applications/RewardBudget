import os
import json
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


# 这里用字符串而不是 tuple，避免 q + INSTRUCTION 报错
INSTRUCTION = (
    # " Based on the question and the given choices, reason step by step, then select the single correct answer from A/B/C/D and output only one of \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D}, with no additional text."
    ""
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="保存 JSONL/Parquet 的目录（必填）",
    )
    return parser.parse_args()


def to_letter(ans):
    """
    mmlu 的 answer 可能是：
    - int: 0/1/2/3
    - str: 'A'/'B'/'C'/'D' 或 '0'/'1'/'2'/'3'
    统一转成 'A'~'D'
    """
    if isinstance(ans, int):
        mapping = ["A", "B", "C", "D"]
        return mapping[ans] if 0 <= ans < 4 else str(ans)

    s = str(ans).strip()
    if s in ["A", "B", "C", "D"]:
        return s
    if s in ["0", "1", "2", "3"]:
        mapping = ["A", "B", "C", "D"]
        return mapping[int(s)]
    return s


def main():
    args = parse_args()
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    output_jsonl = os.path.join(save_dir, "mmlu_pro.jsonl")
    output_parquet = os.path.join(save_dir, "mmlu_pro.parquet")

    # 通常用于评测的是 config="all", split="test"
    print("Loading cais/mmlu (config='all', split='test')...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    records = []
    for ex in ds:
        q = ex.get("question", "").strip()
        ans = ex.get("answer", "")
        gt = to_letter(ans)

        prompt = q + INSTRUCTION

        records.append(
            {
                "prompt": prompt,
                "groundtruth": gt,
            }
        )

    print(f"Total samples: {len(records)}")

    # 写 JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写 Parquet
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_parquet)

    print(f"Saved to:\n  {output_jsonl}\n  {output_parquet}")


if __name__ == "__main__":
    main()
