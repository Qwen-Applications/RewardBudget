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
    # Other nodes wait for training to complete
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
    # Primary node runs training
    export RAY_API_SERVER_ADDRESS="http://localhost:8265"
    echo "提交Dr.GRPO训练脚本 at RANK ${RANK} (${RAY_API_SERVER_ADDRESS})"
    ray job submit --address="${RAY_API_SERVER_ADDRESS}" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        -- \
        python -m verl.trainer.main_ppo \
        data.train_files="${DATA_DIR}/GARO/train_data/${TRAIN_DATA}_train.parquet" \
        data.val_files="${DATA_DIR}/GARO/train_data/${TRAIN_DATA}_valid.parquet" \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        data.train_batch_size=32 \
        actor_rollout_ref.model.path="${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.enable_activation_offload=False \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.adaptive_beta.enable=False \
        actor_rollout_ref.actor.adaptive_beta.beta_min=0.0 \
        actor_rollout_ref.actor.adaptive_beta.beta_max=0.0 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
        actor_rollout_ref.actor.optim.total_training_steps=3000 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.dtype=bfloat16 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.do_sample=True \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.calculate_log_probs=False \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=False \
        custom_reward_function.path=recipe/GARO/reward.py \
        custom_reward_function.name=compute_score \
        algorithm.use_kl_in_reward=False \
        algorithm.adv_estimator=grpo \
        algorithm.ref_sync_freq=20000000 \
        algorithm.norm_adv_by_std_in_grpo=False \
        trainer.log_train_entropy_during_val=True \
        trainer.train_entropy_num_batches=4 \
        trainer.total_epochs=10 \
        trainer.project_name=GARO \
        trainer.experiment_name=${TRAIN_DATA}_DRGRPO \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=1000 \
        trainer.nnodes=${NNODES} \
        trainer.n_gpus_per_node=${NUM_ACCELERATORS} \
        trainer.save_freq=25 \
        trainer.test_freq=5 \
        trainer.default_local_dir="${PRIMUS_SAVE_CHECKPOINT_DIR}"
    
    # Capture exit status
    TRAINING_STATUS=$?

    if [ $TRAINING_STATUS -eq 0 ]; then
        # Only create completion file if training succeeds
        touch ${PRIMUS_OUTPUT_DIR}/training_complete
        echo "Training completed successfully"
    else
        echo "Training failed with exit code $TRAINING_STATUS"
        # Create a failure file to signal other nodes
        echo $TRAINING_STATUS > ${PRIMUS_OUTPUT_DIR}/training_failed
        ray stop
        exit $TRAINING_STATUS
    fi
fi


# Cleanup for all nodes
ray stop
sleep 2h
exit 0
