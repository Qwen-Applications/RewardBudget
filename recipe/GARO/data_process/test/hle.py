import os
import json
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help=f"保存 JSONL/Parquet 的目录（默认: {DEFAULT_SAVE_DIR}）",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="hle",
        help=f"输出文件前缀名（默认: {DEFAULT_DATA_NAME}，会生成 data_name.jsonl / data_name.parquet）",
    )
    return parser.parse_args()


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

        ex_id = ex.get("id", "")
        question = ex.get("question", "")
        answer = ex.get("answer", "")
        answer_type = ex.get("answer_type", "")

        records.append(
            {
                "id": ex_id,
                "prompt": question,
                "data_source": args.data_name,
                "reward_model": {"ground_truth": answer},
                "answer_type": answer_type,
            }
        )

    print(f"Kept {len(records)} text-only samples (image is None).")

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
