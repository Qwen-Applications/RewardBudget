#!/bin/bash
set -e

# 关掉 hf_transfer，加回普通下载，避免 no permits available 那个错误
export HF_HUB_ENABLE_HF_TRANSFER=0

# 你原来的设置
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files=data/OpenMathReasoning_train.parquet \
  data.val_files=data/OpenMathReasoning_valid.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.model.enable_activation_offload=False \
  actor_rollout_ref.model.trust_remote_code=True \
  \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.adaptive_beta.enable=True \
  actor_rollout_ref.actor.adaptive_beta.beta_min=0.0 \
  actor_rollout_ref.actor.adaptive_beta.beta_max=0.005 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.total_training_steps=10 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
  \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.do_sample=True \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.calculate_log_probs=False \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  \
  custom_reward_function.path=recipe/GARO/reward.py \
  custom_reward_function.name=compute_score \
  \
  algorithm.adv_estimator=reward_budget \
  algorithm.ref_sync_freq=200 \
  algorithm.use_kl_in_reward=False \
  \
  trainer.log_train_entropy_during_val=True \
  trainer.train_entropy_num_batches=4 \
  trainer.total_epochs=10 \
  trainer.project_name=GARO \
  trainer.experiment_name=example \
  trainer.logger='["console","wandb"]' \
  trainer.log_val_generations=0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  trainer.save_freq=10 \
  trainer.test_freq=2 \
  trainer.default_local_dir='checkpoints/${trainer.project_name}/${trainer.experiment_name}'
