# Evaluate ${PRIMUS_SAVE_CHECKPOINT_DIR}

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/math500.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_math500.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_math500.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/amc23.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_amc23.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_amc23.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/aime24.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_aime24.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_aime24.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/aime25.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_aime25.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_aime25.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/omni_math.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_omni_math.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_omni_math.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/minerva.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_minerva.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_minerva.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_en_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_en_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_en_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_en_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_en_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_zh_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_zh_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_zh_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_math_zh_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_zh_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_math_zh_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_en_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_en_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_en_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_en_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_en_cee.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_comp.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_zh_comp.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_zh_comp.parquet --k 1

python -m verl.trainer.main_generation \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.path=${PRIMUS_TEST_DATA_PATH}/OlympiadBench_physics_zh_cee.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.output_path=${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_zh_cee.parquet \
  actor_rollout_ref.model.path=${PRIMUS_SAVE_CHECKPOINT_DIR} \
  +actor_rollout_ref.model.trust_remote_code=True \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.prompt_length=1024 \
  rollout.response_length=8192

python3 recipe/GARO/evaluation.py --file ${PRIMUS_OUTPUT_DIR}/reward_budget_mean_adaptive_kl_0.001_0.005_OlympiadBench_physics_zh_cee.parquet --k 1
