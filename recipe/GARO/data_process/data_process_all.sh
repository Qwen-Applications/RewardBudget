# 训练/验证数据集: ${PRIMUS_DATA_PATH}, ${PRIMUS_TEST_DATA_PATH}

python3 recipe/GARO/data_process/train_valid/openmath.py --train_data_dir ${PRIMUS_DATA_PATH} --valid_data_dir ${PRIMUS_TEST_DATA_PATH} --tokenizer=Qwen/Qwen3-8B

python3 recipe/GARO/data_process/train_valid/openscience.py --train_data_dir ${PRIMUS_DATA_PATH} --valid_data_dir ${PRIMUS_TEST_DATA_PATH} --tokenizer=Qwen/Qwen3-8B

# 测试数据集: ${PRIMUS_OTHER_DATA_PATH}

python3 recipe/GARO/data_process/test/general.py --save_dir ${PRIMUS_OTHER_DATA_PATH}

python3 recipe/GARO/data_process/test/hle.py --save_dir ${PRIMUS_OTHER_DATA_PATH}

python3 recipe/GARO/data_process/test/mmlu_pro.py --save_dir ${PRIMUS_OTHER_DATA_PATH}

python3 recipe/GARO/data_process/test/olympiadbench.py --save_dir ${PRIMUS_OTHER_DATA_PATH}