#!/bin/bash
set -x

# ====== 多机必要环境变量检查 ======
# 在每台机器上事先 export：
#   MASTER_ADDR=<主节点IP>
#   MASTER_PORT=<主节点端口，例如 29500>
#   NODE_RANK=0/1/2/3
if [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$NODE_RANK" ]; then
  echo "Please set MASTER_ADDR, MASTER_PORT, NODE_RANK before running."
  exit 1
fi

# 每台机器 8 卡，总共 4 台
NNODES=4
NPROC_PER_NODE=8
PRIMUS_DATA_PATH=
PRIMUS_TEST_DATA_PATH=
PRIMUS_SAVE_CHECKPOINT_DIR=需要指向具体的ckpt地址，比如/global_step_1000

# 你想存 checkpoint 的总目录，外面先 export PRIMUS_SAVE_CHECKPOINT_DIR=/some/path
if [ -z "$PRIMUS_SAVE_CHECKPOINT_DIR" ]; then
  echo "Please set PRIMUS_SAVE_CHECKPOINT_DIR"
  exit 1
fi

torchrun \
  --nnodes=${NNODES} \
  --nproc_per_node=${NPROC_PER_NODE} \
  --node_rank=${NODE_RANK} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files=${PRIMUS_DATA_PATH}/OpenMathReasoning_train.parquet \
  data.val_files=${PRIMUS_DATA_PATH}/OpenMathReasoning_valid.parquet \
  data.prompt_key=prompt \
  data.response_key=response \
  data.micro_batch_size_per_gpu=1 \
  model.partial_pretrain=Qwen/Qwen3-8B \
  model.enable_gradient_checkpointing=True \
  model.trust_remote_code=True \
  trainer.project_name=GARO \
  trainer.experiment_name=OpenMath_SFT \
  trainer.total_epochs=5 \
  trainer.default_local_dir=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  trainer.logger='["console","wandb"]'


# Evaluate ${PRIMUS_SAVE_CHECKPOINT_DIR}

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/math500.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_math500.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_math500.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/amc23.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_amc23.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_amc23.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/aime24.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_aime24.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_aime24.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/aime25.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/aime25.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_aime25.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/omni_math.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_omni_math.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_omni_math.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/minerva.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_minerva.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_minerva.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_en_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_en_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_en_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_en_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_zh_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_zh_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_zh_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_zh_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_math_zh_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_en_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_en_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_en_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_en_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_zh_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_zh_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_zh_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/sft_OlympiadBench_physics_zh_cee.parquet --k 1
