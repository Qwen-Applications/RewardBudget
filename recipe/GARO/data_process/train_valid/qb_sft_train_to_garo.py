#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRAIN_IN = Path("/primus_datasets/zmy/QB-SFT/train_data/OpenScienceReasoning2_train.parquet")
VALID_IN = Path("/primus_datasets/zmy/QB-SFT/train_data/OpenScienceReasoning2_valid.parquet")

TRAIN_OUT = Path("/primus_datasets/zmy/GARO/train_data/OpenScienceReasoning2_train.parquet")
VALID_OUT = Path("/primus_datasets/zmy/GARO/train_data/OpenScienceReasoning2_valid.parquet")

QB_SFT_ROOT = Path("/primus_datasets/zmy/QB-SFT").resolve()
GARO_ROOT = Path("/primus_datasets/zmy/GARO").resolve()


def ensure_safe_paths(input_path: Path, output_path: Path) -> None:
    input_resolved = input_path.resolve()
    output_parent_resolved = output_path.parent.resolve()

    if not str(input_resolved).startswith(str(QB_SFT_ROOT)):
        raise ValueError(f"输入路径不在 QB-SFT 下，拒绝执行: {input_path}")

    if not str(output_parent_resolved).startswith(str(GARO_ROOT)):
        raise ValueError(f"输出路径不在 GARO 下，拒绝执行: {output_path}")

    if input_resolved == output_path.resolve():
        raise ValueError(f"输入和输出路径相同，拒绝覆盖原数据: {input_path}")

    if str(output_path.resolve()).startswith(str(QB_SFT_ROOT)):
        raise ValueError(f"输出路径位于 QB-SFT 下，拒绝写入: {output_path}")


def extract_ground_truth(reward_model):
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth", "")

    # 兼容少数 parquet 读出来可能不是普通 dict 的情况
    try:
        return reward_model["ground_truth"]
    except Exception:
        return ""


def convert_one_file(input_path: Path, output_path: Path) -> None:
    ensure_safe_paths(input_path, output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    print(f"Reading:  {input_path}")
    df = pd.read_parquet(input_path)

    required_columns = {"prompt", "reward_model"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} 缺少必要列: {missing}")

    out_df = pd.DataFrame()
    out_df["prompt"] = df["prompt"].apply(
        lambda x: [{"role": "user", "content": "" if pd.isna(x) else str(x)}]
    )
    out_df["response"] = ""
    out_df["reward_model"] = df["reward_model"].apply(
        lambda rm: {"ground_truth": extract_ground_truth(rm)}
    )
    out_df["ability"] = "math"
    out_df["data_source"] = "dapo-math-17k"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写临时文件，再原子替换，避免中途失败留下半个 parquet
    with tempfile.NamedTemporaryFile(
        dir=str(output_path.parent),
        suffix=".parquet",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    print(f"Written:  {output_path}")
    print(f"Rows:     {len(out_df)}")


def main():
    convert_one_file(TRAIN_IN, TRAIN_OUT)
    convert_one_file(VALID_IN, VALID_OUT)


if __name__ == "__main__":
    main()