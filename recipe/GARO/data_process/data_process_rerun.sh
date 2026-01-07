TEST_DIR="${PRIMUS_OSS_WORKSPACE_DIR}/GARO/test_data/"
python recipe/GARO/data_process/test/hle.py --save_dir ${TEST_DIR}
python recipe/GARO/data_process/test/mmlu_pro.py --save_dir ${TEST_DIR}
python recipe/GARO/data_process/test/olympiadbench.py --save_dir ${TEST_DIR}
