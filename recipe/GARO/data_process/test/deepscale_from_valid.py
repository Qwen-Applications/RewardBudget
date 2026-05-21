import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd


LOCAL_TMP_ROOT = Path("/root/tmp")
LOCAL_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def safe_to_parquet_via_tmp(df: pd.DataFrame, output_path: str):
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 先写到 /root/tmp/，避免 pandas/pyarrow 直接写网盘
    with tempfile.TemporaryDirectory(dir=LOCAL_TMP_ROOT) as tmp_dir:
        local_tmp_path = Path(tmp_dir) / output_path.name
        df.to_parquet(local_tmp_path, index=False)

        if local_tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Local parquet is empty: {local_tmp_path}")

        # 2. 再移动到目标目录下的临时文件
        remote_tmp_path = output_dir / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
        shutil.move(str(local_tmp_path), str(remote_tmp_path))

        if remote_tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Remote tmp parquet is empty: {remote_tmp_path}")

        # 3. 原子替换最终文件
        os.replace(remote_tmp_path, output_path)


src_path = "/primus_datasets/zmy/GARO/train_data/deepscale_valid.parquet"
dst_path = "/primus_datasets/zmy/GARO/test_data/deepscale.parquet"

df = pd.read_parquet(src_path)

keep_cols = ["prompt", "reward_model", "data_source"]

if "data_source" not in df.columns:
    df["data_source"] = "deepscale"

missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in source parquet: {missing}")

test_df = df[keep_cols].copy()

safe_to_parquet_via_tmp(test_df, dst_path)

print(f"Saved test data to: {dst_path}")
print(f"Rows: {len(test_df)}")
print(f"Columns: {list(test_df.columns)}")
print(f"File size: {os.path.getsize(dst_path)} bytes")