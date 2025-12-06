#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
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

def prompt_to_content(p):
    if isinstance(p, str):
        return p
    try:
        return json.dumps(p, ensure_ascii=False)
    except Exception:
        return str(p)

def list_parquet_files(path: str):
    # 关键修复：兼容 “xxx.parquet 是目录”的情况
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        return files
    return [path]

def main():
    test_dir = os.path.join(DATA_DIR, "GARO", "test_data")

    extra_type = pa.list_(
        pa.struct([
            ("role", pa.string()),
            ("content", pa.string()),
        ])
    )

    for task in TASKS:
        path = os.path.join(test_dir, f"{task}.parquet")
        if not os.path.exists(path):
            print(f"[SKIP] not found: {path}")
            continue

        shard_files = list_parquet_files(path)
        if not shard_files:
            print(f"[SKIP] no parquet shards under: {path}")
            continue

        for fp in shard_files:
            # 用 ParquetFile 读单文件，避免 dataset unify
            table = pq.ParquetFile(fp).read()

            if "prompt" not in table.column_names:
                print(f"[WARN] no prompt col: {fp}")
                continue

            # 如果你非常确定源文件都没有 extra_info，这个判断可删
            if "extra_info" in table.column_names:
                print(f"[SKIP] already has extra_info: {fp}")
                continue

            prompts = table["prompt"].to_pylist()
            extras = [
                [{"role": "user", "content": prompt_to_content(p)}]
                for p in prompts
            ]

            table = table.append_column("extra_info", pa.array(extras, type=extra_type))

            # 直接覆盖 shard 文件
            pq.write_table(table, fp)
            print(f"[OK] overwritten: {fp}")

    print("[DONE]")

if __name__ == "__main__":
    main()
