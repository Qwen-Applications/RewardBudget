#!/bin/bash
set -e

export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
DATA_DIR="/primus_datasets/zmy"
RAY_PORT=6379

# Quit the script if ${RANK} is not 0
if [ "${RANK}" -ne 0 ]; then
    echo "RANK is ${RANK}, not 0, skip the training script."
    exit 0
fi

export RAY_API_SERVER_ADDRESS="http://${MASTER_ADDR}:${RAY_PORT}"
echo "提交GRPO训练脚本 at RANK ${RANK} (${RAY_API_SERVER_ADDRESS})"
ray job submit --address="${RAY_API_SERVER_ADDRESS}" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    -- \
    python -m verl.trainer.main_ppo \
      data.train_files="${DATA_DIR}/GARO/train_data/OpenMathReasoning_train.parquet" \
      data.val_files="${DATA_DIR}/GARO/train_data/OpenMathReasoning_valid.parquet" \
      data.max_prompt_length=1024 \
      data.max_response_length=8192 \
      data.train_batch_size=128 \
      actor_rollout_ref.model.path=${PRIMUS_SOURCE_CHECKPOINT_DIR} \
      actor_rollout_ref.model.enable_gradient_checkpointing=False \
      actor_rollout_ref.model.enable_activation_offload=False \
      actor_rollout_ref.model.trust_remote_code=True \
      actor_rollout_ref.actor.strategy=fsdp \
      actor_rollout_ref.actor.ppo_mini_batch_size=256 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.actor.use_dynamic_bsz=False \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.use_kl_in_reward=False \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.adaptive_beta.enable=False \
      actor_rollout_ref.actor.adaptive_beta.beta_min=0.0 \
      actor_rollout_ref.actor.adaptive_beta.beta_max=0.005 \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
      actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
      actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
      actor_rollout_ref.actor.optim.total_training_steps=2000 \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
      rollout.name=vllm \
      rollout.temperature=1.0 \
      rollout.top_k=-1 \
      rollout.top_p=1.0 \
      rollout.dtype=bfloat16 \
      rollout.gpu_memory_utilization=0.5 \
      rollout.log_prob_micro_batch_size_per_gpu=16 \
      rollout.do_sample=True \
      rollout.n=8 \
      rollout.calculate_log_probs=False \
      rollout.val_kwargs.top_k=-1 \
      rollout.val_kwargs.top_p=1.0 \
      rollout.val_kwargs.temperature=0.0 \
      rollout.val_kwargs.n=1 \
      rollout.val_kwargs.do_sample=False \
      custom_reward_function.path=recipe/GARO/reward.py \
      custom_reward_function.name=compute_score \
      algorithm.adv_estimator=grpo \
      algorithm.ref_sync_freq=20000000 \
      trainer.log_train_entropy_during_val=True \
      trainer.train_entropy_num_batches=4 \
      trainer.total_epochs=10 \
      trainer.project_name=GARO \
      trainer.experiment_name=OpenMath_GRPO \
      trainer.logger='["console","wandb"]' \
      trainer.log_val_generations=0 \
      trainer.nnodes=${NNODES} \
      trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
      trainer.save_freq=200 \
      trainer.test_freq=200 \
      trainer.default_local_dir="${PRIMUS_SAVE_CHECKPOINT_DIR}"

# # Evaluate ${PRIMUS_SAVE_CHECKPOINT_DIR}

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/math500.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_math500.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_math500.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/amc23.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_amc23.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_amc23.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/aime24.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_aime24.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_aime24.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/aime25.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/aime25.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_aime25.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/omni_math.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_omni_math.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_omni_math.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/minerva.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_minerva.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_minerva.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_comp.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_en_comp.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_en_comp.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_cee.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_en_cee.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_en_cee.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_cee.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_zh_comp.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_zh_comp.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_zh_cee.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_zh_cee.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_math_zh_cee.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_comp.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_en_comp.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_en_comp.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_cee.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_en_cee.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_en_cee.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_comp.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_zh_comp.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_zh_comp.parquet --k 1

# python -m verl.trainer.main_generation \
#   trainer.nnodes=4 \
#   trainer.n_gpus_per_node=8 \
#   data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_cee.parquet \
#   data.prompt_key=prompt \
#   data.n_samples=1 \
#   data.output_path=${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_zh_cee.parquet \
#   actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
#   +actor_rollout_ref.model.trust_remote_code=True \
#   rollout.temperature=0.0 \
#   rollout.top_p=1.0 \
#   rollout.prompt_length=1024 \
#   rollout.response_length=8192

# python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/grpo_OlympiadBench_physics_zh_cee.parquet --k 1

sleep 2h
