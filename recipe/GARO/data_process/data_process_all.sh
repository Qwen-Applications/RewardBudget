# 训练/验证数据集: ${PRIMUS_OSS_WORKSPACE_DIR}/GARO/train_data/
TRAIN_DIR="${PRIMUS_OSS_WORKSPACE_DIR}/GARO/train_data/"
python recipe/GARO/data_process/train_valid/dapo_math_17k.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=Qwen/Qwen3-8B
python recipe/GARO/data_process/train_valid/openmath.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=Qwen/Qwen3-8B
python recipe/GARO/data_process/train_valid/openscience.py --train_data_dir ${TRAIN_DIR} --valid_data_dir ${TRAIN_DIR} --tokenizer=Qwen/Qwen3-8B

# 测试数据集: ${PRIMUS_OSS_WORKSPACE_DIR}/GARO/test_data/
TEST_DIR="${PRIMUS_OSS_WORKSPACE_DIR}/GARO/test_data/"
python recipe/GARO/data_process/test/general.py --save_dir ${TEST_DIR}
python recipe/GARO/data_process/test/hle.py --save_dir ${TEST_DIR}
python recipe/GARO/data_process/test/mmlu_pro.py --save_dir ${TEST_DIR}
python recipe/GARO/data_process/test/olympiadbench.py --save_dir ${TEST_DIR}
