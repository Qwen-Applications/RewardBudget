#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_TOKEN=REDACTED_HF_TOKEN
# export OUTPUT_DIR="${PRIMUS_OSS_WORKSPACE_DIR}"
export OUTPUT_DIR=/primus_datasets/zmy
echo "[INFO] HF_ENDPOINT=${HF_ENDPOINT}"
echo "[INFO] HF_TOKEN=${HF_TOKEN}"

# 训练/验证数据集: ${OUTPUT_DIR}/GARO/train_data/
TRAIN_DIR="${OUTPUT_DIR}/GARO/train_data/"
# python recipe/GARO/data_process/train_valid/dapo_math_17k.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=Qwen/Qwen3-8B
# python recipe/GARO/data_process/train_valid/openmath.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=Qwen/Qwen3-8B
python recipe/GARO/data_process/train_valid/openscience.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=${PRIMUS_SOURCE_CHECKPOINT_DIR}/${PRIMUS_SOURCE_CHECKPOINT_ITERATION_DIRNAME} --china

# 测试数据集: ${OUTPUT_DIR}/GARO/test_data/
# TEST_DIR="${OUTPUT_DIR}/GARO/test_data/"
# python recipe/GARO/data_process/test/general.py --save_dir ${TEST_DIR}
# python recipe/GARO/data_process/test/hle.py --save_dir ${TEST_DIR}
# python recipe/GARO/data_process/test/mmlu_pro.py --save_dir ${TEST_DIR}
# python recipe/GARO/data_process/test/olympiadbench.py --save_dir ${TEST_DIR}
