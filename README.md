<div align="center">
<h1>RewardBudget</h1>

<!-- Badges -->
<a href="TODO">
<img src="https://img.shields.io/badge/arXiv-TODO-b31b1b.svg?style=for-the-badge" alt="arXiv">
</a>
<a href="TODO">
<img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github">
</a>
<a href="https://opensource.org/licenses/Apache-2.0">
<img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License">
</a>

<p align="center">
<i><b>TODO: team / affiliation</b></i>
</p>

RewardBudget is a verl-based RL training recipe for reasoning models. To use the method, set
<code>algorithm.adv_estimator=reward_budget</code>. The released code keeps the core estimator,
the boxed-answer reward function, and concise train/evaluation entrypoints.
</div>

## 1. Setup

```bash
git clone TODO
cd RewardBudget
python -m pip install -e .
```

Install the rollout backend you plan to use, such as vLLM or SGLang, following the upstream verl environment requirements for your CUDA/PyTorch stack.

## 2. Data Format

Prepare train/validation/evaluation data as verl-compatible parquet files:

| Column | Type | Description |
|---|---|---|
| `prompt` | list of chat messages | Usually `[{"role": "user", "content": "..."}]` |
| `reward_model` | dict | Must include `{"ground_truth": "..."}` |
| `data_source` | string | Dataset name used by reward/evaluation routing |
| `response` | string, optional | Needed only for SFT-style warm starts |

For the default reward in `recipe/RewardBudget/reward.py`, model outputs should put the final answer in `\boxed{...}`.

## 3. Train

Run the example training script:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
TRAIN_FILE=data/train.parquet \
VAL_FILE=data/valid.parquet \
SAVE_DIR=checkpoints/RewardBudget/qwen2p5_7b \
bash recipe/RewardBudget/scripts/train/run_reward_budget.sh
```

The essential switch is:

```bash
algorithm.adv_estimator=reward_budget
```

You can override any verl/Hydra option by appending it to the command:

```bash
bash recipe/RewardBudget/scripts/train/run_reward_budget.sh \
  data.train_batch_size=64 \
  actor_rollout_ref.rollout.n=16
```

## 4. Evaluate

Generate responses and compute pass@k:

```bash
bash recipe/RewardBudget/scripts/evaluate/run_eval.sh \
  checkpoints/RewardBudget/qwen2p5_7b/global_step_100/actor \
  data/test.parquet \
  outputs/eval_reward_budget
```

For exact-match math-style tasks, the evaluator uses the local boxed-answer reward. For other data sources, `recipe/RewardBudget/evaluation.py` can call an OpenAI-compatible completions endpoint:

```bash
python recipe/RewardBudget/evaluation.py \
  --file outputs/eval_reward_budget/generations.parquet \
  --k 1 \
  --output outputs/eval_reward_budget/eval.json \
  --output_jsonl outputs/eval_reward_budget/eval.details.jsonl \
  --qwen_text_model_url http://localhost:8000/v1/completions
```

## 5. Repository Structure

```text
RewardBudget/
|-- README.md
|-- recipe/RewardBudget/
|   |-- reward.py                         # Boxed-answer reward
|   |-- evaluation.py                     # pass@k evaluator
|   `-- scripts/
|       |-- train/run_reward_budget.sh    # Training example
|       `-- evaluate/run_eval.sh          # Evaluation example
|-- verl/trainer/ppo/core_algos.py        # RewardBudget estimator registration
`-- verl/                                 # Upstream verl training stack
```

## 6. Notes

- This repository is based on verl and keeps the upstream Apache 2.0 license.
- Paper URL, final GitHub URL, released checkpoints, and exact reproduction dataset links are TODO.
- Historical experimental branches, adaptive beta, and periodic reference-model sync are not part of the release path.

## Citation

```bibtex
TODO
```
