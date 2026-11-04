<div align="center">
<h1>[NeurIPS2026] Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning</h1>

<!-- Badges -->
<a href="https://github.com/Qwen-Applications">
  <img src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge" alt="Qwen Applications">
</a>
<a href="2611.TODO">
<img src="https://img.shields.io/badge/arXiv-2611.TODO-b31b1b.svg?style=for-the-badge" alt="arXiv">
</a>
<a href="https://github.com/Qwen-Applications/RewardBudget">
<img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github">
</a>
<a href="https://opensource.org/licenses/Apache-2.0">
<img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License">
</a>

<p align="center">
  <i><b>Qwen Large Model Application Team, Alibaba</b></i>
</p>

RewardBudget is a plug-and-play RL recipe for LLM reasoning that mitigates premature
convergence by regulating algorithm-induced sharpening rather than explicitly preserving entropy.
It assigns each prompt a fixed reward budget and redistributes that budget among correct rollouts,
producing gentler update dynamics and an approximately log-shaped objective. This repository
provides the verl estimator, boxed-answer reward function, and concise train/evaluation entrypoints
for reproducing RewardBudget experiments.

</div>

## 1. Setup

```bash
git clone https://github.com/Qwen-Applications/RewardBudget
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

Generate responses and compute pass@k (here with `K=8` sampled responses per question):

```bash
K=8 bash recipe/RewardBudget/scripts/evaluate/run_eval.sh \
  checkpoints/RewardBudget/qwen2p5_7b/global_step_100/actor \
  data/test.parquet \
  outputs/eval_reward_budget
```

`K` defaults to 1 (greedy decoding, temperature 0). For `K>1`, the script defaults to sampling at temperature 1.0; override with `GEN_TEMPERATURE` if needed. The generation overrides use the `model.*` and `rollout.*` keys from `verl/trainer/config/generation.yaml`.

Local boxed-answer string matching is used when `data_source` is `OpenMath`, `dapo-math-17k`, `amc23`, `aime24`, `aime25`, `math500`, `gpqa_diamond`, `gpqa_main`, `gpqa_extended`, or `mmlu_pro`. Matching strips surrounding whitespace and collapses runs of internal whitespace but **does not check mathematical equivalence**. Other data sources require a running OpenAI-compatible completions judge; failed judge requests are scored as incorrect, so check the endpoint and evaluation logs before trusting those metrics:

```bash
python -m recipe.RewardBudget.evaluation \
  --file outputs/eval_reward_budget/generations.parquet \
  --k 8 \
  --output outputs/eval_reward_budget/eval.json \
  --output_jsonl outputs/eval_reward_budget/eval.details.jsonl \
  --judge_url http://localhost:8000/v1/completions
```

The built-in judge prompt uses Qwen3-Instruct (ChatML) formatting. Adapt `apply_chat_template` in `evaluation.py` if your judge model requires a different prompt format. The legacy `--qwen_*` judge flags are still accepted.

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
@article{jia2026rewardbudgeting,
  title={Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning},
  author={Mengni Jia, Mengyu Zhou, Xiaoxi Jiang, Guanjun Jiang},
  journal={arXiv preprint arXiv:2611.TODO},
  year={2026}
}
```
