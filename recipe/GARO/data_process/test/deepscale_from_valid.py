import os
import pandas as pd

src_path = "/primus_datasets/zmy/GARO/train_data/deepscale_valid.parquet"
dst_path = "/primus_datasets/zmy/GARO/test_data/deepscale.parquet"

os.makedirs(os.path.dirname(dst_path), exist_ok=True)

df = pd.read_parquet(src_path)

# test data 需要的核心字段
keep_cols = ["prompt", "reward_model", "data_source"]

# 如果原数据没有 data_source，就补一个
if "data_source" not in df.columns:
    df["data_source"] = "deepscale"

# 只保留 test data 格式字段
missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in source parquet: {missing}")

test_df = df[keep_cols].copy()

# 保存为新的 test parquet，不影响原 valid parquet
test_df.to_parquet(dst_path, index=False)

print(f"Saved test data to: {dst_path}")
print(f"Rows: {len(test_df)}")
print(f"Columns: {list(test_df.columns)}")