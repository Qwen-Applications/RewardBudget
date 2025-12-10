#!/bin/bash
set -e

export VLLM_USE_NCCL_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
DATA_DIR="/primus_datasets/zmy"
TRAIN_DATA="dapo_math_17k"  # "OpenMathReasoning"

if [ "${RANK}" -ne 0 ]; then
    # 其他节点只等待 0 号节点训练
    echo "Node ${RANK} waiting for training to complete..."
    while [ ! -f ${PRIMUS_OUTPUT_DIR}/training_complete ] && [ ! -f ${PRIMUS_OUTPUT_DIR}/training_failed ]; do
        sleep 120
    done

    if [ -f ${PRIMUS_OUTPUT_DIR}/training_failed ]; then
        FAILED_STATUS=$(cat ${PRIMUS_OUTPUT_DIR}/training_failed)
        echo "Node ${NODE_RANK} detected training failure with status $FAILED_STATUS"
        ray stop
        exit $FAILED_STATUS
    fi

    echo "Node ${NODE_RANK} detected training completion"
else
    # 0 号节点提交 ray job
    export RAY_API_SERVER_ADDRESS="http://localhost:8265"
    echo "提交 CLIP-COV 训练脚本 at RANK ${RANK} (${RAY_API_SERVER_ADDRESS})"

    ray job submit --address="${RAY_API_SERVER_ADDRESS}" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python -m recipe.entropy.main_entropy \
        data.train_files="${DATA_DIR}/GARO/train_data/${TRAIN_DATA}_train.parquet" \
        data.val_files="${DATA_DIR}/GARO/train_data/${TRAIN_DATA}_valid.parquet" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        data.train_batch_size=32 \
        data.gen_batch_size=96 \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.filter_overlong_prompts=False \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.enable_activation_offload=False \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.optim.total_training_steps=3000 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
        actor_rollout_ref.actor.clip_ratio_low=1 \
        actor_rollout_ref.actor.clip_ratio_high=1 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.policy_loss.loss_mode="clip_cov" \
        actor_rollout_ref.actor.policy_loss.clip_cov_ratio=0.0002 \
        actor_rollout_ref.actor.policy_loss.clip_cov_lb=1.0 \
        actor_rollout_ref.actor.policy_loss.clip_cov_ub=5.0 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.do_sample=True \
        actor_rollout_ref.rollout.calculate_log_probs=False \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False \
        custom_reward_function.path=recipe/GARO/reward.py \
        custom_reward_function.name=compute_score \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=False \
        reward_model.overlong_buffer.len=2048 \
        reward_model.overlong_buffer.penalty_factor=1.0 \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.ref_sync_freq=20000000 \
        algorithm.filter_groups.enable=True \
        algorithm.filter_groups.metric=acc \
        algorithm.filter_groups.max_num_gen_batches=10 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name=GARO \
        trainer.experiment_name=${TRAIN_DATA}_CLIPCov \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
        trainer.save_freq=25 \
        trainer.test_freq=5 \
        trainer.total_epochs=10 \
        trainer.default_local_dir="${PRIMUS_SAVE_CHECKPOINT_DIR}" \
        trainer.log_train_entropy_during_val=True \
        trainer.train_entropy_num_batches=4 \
        trainer.log_val_generations=1000


    TRAINING_STATUS=$?

    if [ $TRAINING_STATUS -eq 0 ]; then
        touch ${PRIMUS_OUTPUT_DIR}/training_complete
        echo "Training completed successfully"
    else
        echo "Training failed with exit code $TRAINING_STATUS"
        echo $TRAINING_STATUS > ${PRIMUS_OUTPUT_DIR}/training_failed
        ray stop
        exit $TRAINING_STATUS
    fi
fi

ray stop
sleep 2h
exit 0
