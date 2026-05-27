<div align="center">
<h1>Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning</h1>

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

RewardBudget is a plug-and-play RL recipe for LLM reasoning that mitigates premature
convergence by regulating algorithm-induced sharpening rather than explicitly preserving entropy.
It assigns each prompt a fixed reward budget and redistributes that budget among correct rollouts,
producing gentler update dynamics and an approximately log-shaped objective. This repository
provides the verl estimator, boxed-answer reward function, and concise train/evaluation entrypoints
for reproducing RewardBudget experiments.

## 1. Setup

```bash
git clone TODO
cd RewardBudget
python -m pip install -e .
```

Install the rollout backend you plan to use, such as vLLM or SGLang, following the upstream verl environment requirements for your CUDA/PyTorch stack.

## 2. Data Format

Prepare train/validation/evaluation data as verl-compatible parquet files. Each row should use the following schema:

| Column | Type | Description |
|---|---|---|
| `prompt` | list of chat messages | A chat-style prompt, usually one user message. |
| `response` | string, optional | Existing model response or an empty string for SFT-style warm starts. |
| `reward_model` | dict | Must include `{"ground_truth": "..."}`. |
| `data_source` | string | Dataset name used by reward/evaluation routing, such as `OpenMath`, `amc23`, or `math500`. |
| `ability` | string | Task category, such as `math` or `physics`. |

Example row:

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}. What is 1+1?"
    }
  ],
  "response": "\nOkay, let's see. I need to find the sum of 1 and 1. The answer is \\boxed{2}.",
  "reward_model": {"ground_truth": "2"},
  "data_source": "OpenMath",
  "ability": "math"
}
```

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

## Citation

```bibtex
TODO
```
