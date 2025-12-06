#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
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
    # 你要用 data["prompt"] 填 content
    # 为了避免 prompt 不是 str 时的类型问题，转成字符串
    if isinstance(p, str):
        return p
    try:
        return json.dumps(p, ensure_ascii=False)
    except Exception:
        return str(p)

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

        table = pq.read_table(path)

        if "prompt" not in table.column_names:
            print(f"[WARN] no prompt col: {path}")
            continue

        prompts = table["prompt"].to_pylist()
        extras = [
            [{"role": "user", "content": prompt_to_content(p)}]
            for p in prompts
        ]

        table = table.append_column("extra_info", pa.array(extras, type=extra_type))

        # 直接覆盖
        pq.write_table(table, path)
        print(f"[OK] overwritten: {path}")

    print("[DONE]")

if __name__ == "__main__":
    main()
