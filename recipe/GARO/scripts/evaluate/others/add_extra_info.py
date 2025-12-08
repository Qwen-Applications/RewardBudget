#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "/primus_datasets/zmy"
TASKS = [
    "math500",
    "amc23",
    "aime24",
    "aime25",
    "omni_math",
    "minerva",
    "OlympiadBench_math_en_comp",
    "OlympiadBench_math_en_cee",
    "OlympiadBench_math_zh_comp",
    "OlympiadBench_math_zh_cee",
    "OlympiadBench_physics_en_comp",
    "OlympiadBench_physics_en_cee",
    "OlympiadBench_physics_zh_comp",
    "OlympiadBench_physics_zh_cee",
]

def main():
    parser = argparse.ArgumentParser()
    # new-changes: make input optional for batch mode
    parser.add_argument("--input", "-i", required=False, help="Path to input .jsonl")
    parser.add_argument("--output", "-o", default=None, help="Path to output .parquet")
    args = parser.parse_args()

    def process_one(in_path: str, out_path: str):
        if os.path.isdir(out_path):
            raise ValueError(
                f"Output path is a directory, not a file: {out_path}\n"
                f"Please provide a file path like xxx.parquet"
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        records = []
        with open(in_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSON at line {line_no}: {e}")

                prompt = obj.get("prompt", "")
                # 保证 content 是 string
                if not isinstance(prompt, str):
                    prompt = json.dumps(prompt, ensure_ascii=False)

                # 直接新增/覆盖 extra_info
                obj["extra_info"] = [{"role": "user", "content": prompt}]

                records.append(obj)

        table = pa.Table.from_pylist(records)

        # 直接覆盖写
        pq.write_table(table, out_path)

        print(f"[OK] {in_path} -> {out_path}, rows={table.num_rows}")

    # new-changes: batch mode when --input is not provided
    if not args.input:
        test_dir = os.path.join(DATA_DIR, "GARO", "test_data")
        for task in TASKS:
            in_path = os.path.join(test_dir, f"{task}.jsonl")
            out_path = os.path.join(test_dir, f"{task}.parquet")

            if not os.path.exists(in_path):
                print(f"[SKIP] not found: {in_path}")
                continue

            process_one(in_path, out_path)

        print("[DONE]")
        return
    # end new-changes

    # 原单文件逻辑保持不变
    in_path = args.input
    if args.output:
        out_path = args.output
    else:
        # 默认同名输出：xxx.jsonl -> xxx.parquet
        base, _ = os.path.splitext(in_path)
        out_path = base + ".parquet"

    process_one(in_path, out_path)


if __name__ == "__main__":
    main()
