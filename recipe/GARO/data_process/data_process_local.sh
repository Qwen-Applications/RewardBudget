# 训练/验证数据集: "train_data"
python train_valid/openmath.py --train_data_dir "train_data" --valid_data_dir "train_data" --tokenizer=Qwen/Qwen3-8B
python train_valid/openscience.py --train_data_dir "train_data" --valid_data_dir "train_data" --tokenizer=Qwen/Qwen3-8B

# 测试数据集: "test_data"
python test/general.py --save_dir "test_data"
python test/hle.py --save_dir "test_data"
python test/mmlu_pro.py --save_dir "test_data"
python test/olympiadbench.py --save_dir "test_data"
