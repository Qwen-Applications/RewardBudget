# recipe/GARO/eval_pass_k.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Dict

import torch
import pyarrow.parquet as pq

from recipe.GARO.reward import compute_score

# =====================
# 1) data_source routing
# =====================
LOCAL_SOURCES = {
    "gpqa_diamond",
    "gpqa_main",
    "gpqa_extended",
    "mmlu_pro",
    "amc23",
    "aime24",
    "aime25",
}

# =====================
# 2) Utils
# =====================
def is_main_process() -> bool:
    """只在未初始化 DDP 或 rank0 时返回 True。"""
    return (not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0)


def local_mkdir(path: str) -> str:
    """只在 main process 创建目录，然后所有 rank 同步一下。"""
    if path == "":
        return path

    if is_main_process():
        os.makedirs(path, exist_ok=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return path


def _iter_records(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    # --- JSONL / JSON ---
    if ext in [".jsonl", ".json"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    # --- Parquet ---
    elif ext == ".parquet":
        table = pq.read_table(file_path)
        cols = table.to_pydict()
        n = table.num_rows

        for i in range(n):
            row = {k: v[i] for k, v in cols.items()}
            yield row
    else:
        raise ValueError(f"Unsupported file extension for {file_path}")


# =====================
# 3) A lightweight boxed extractor
# =====================
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def extract_solution_inside_boxed(text: str) -> str:
    """
    简易版：抽取最后一个 \boxed{...} 里的内容。
    如果没有 boxed，就返回空串。
    """
    if not text:
        return ""
    matches = _BOXED_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip()


# =====================
# 4) API Judge (template)
# =====================
@dataclass
class JudgeConfig:
    api_base: str
    api_key: str
    model: str = "gpt-4o-mini"  # 你可以改成你们的 judge 模型
    timeout: int = 60
    temperature: float = 0.0


def build_judge_prompt(question: str, ground_truth: str, pred_answer: str) -> str:
    return f"""You are a strict evaluator.

You will be given:
1) The problem.
2) The ground-truth answer.
3) The model's extracted final answer.

Your job is to decide whether the model's answer is correct.

Rules:
- Judge correctness based on the problem and the ground-truth answer.
- Ignore any reasoning. Only evaluate the final answer.
- If the answer is correct, reply with \\boxed{{correct}}.
- Otherwise, reply with \\boxed{{incorrect}}.
- Do NOT output anything else.

Problem:
{question}

Ground-truth answer:
{ground_truth}

Model's extracted final answer:
{pred_answer}
"""


def parse_judge_boxed(output_text: str) -> Optional[bool]:
    """
    解析 judge 模型输出：
    - \boxed{correct} -> True
    - \boxed{incorrect} -> False
    解析失败返回 None
    """
    if not output_text:
        return None
    m = _BOXED_RE.search(output_text)
    if not m:
        # 容错：万一模型没 boxed
        lowered = output_text.strip().lower()
        if lowered == "correct":
            return True
        if lowered == "incorrect":
            return False
        return None
    tag = m.group(1).strip().lower()
    if tag == "correct":
        return True
    if tag == "incorrect":
        return False
    return None


def call_judge_api(prompt: str, cfg: JudgeConfig) -> str:
    """
    通用 HTTP 模板（你需要按你们的 API 规范改字段）。
    如果你们用的是 OpenAI-compatible 接口，这个结构通常能对上。
    """
    import requests

    url = cfg.api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [
            {"role": "system", "content": "You are a precise grading assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI-compatible parsing
    return data["choices"][0]["message"]["content"]


def api_judge_score(question: str, ground_truth: str, solution_str: str, cfg: JudgeConfig) -> float:
    """
    使用 API judge 判断 correctness。
    返回 1.0/0.0
    """
    pred_answer = extract_solution_inside_boxed(solution_str)
    prompt = build_judge_prompt(question, ground_truth, pred_answer)

    # 重要：如果你真的在 DDP 环境跑评测，避免多 rank 重复扣费
    if torch.distributed.is_initialized() and not is_main_process():
        return 0.0

    try:
        out = call_judge_api(prompt, cfg)
        verdict = parse_judge_boxed(out)
        if verdict is None:
            return 0.0
        return 1.0 if verdict else 0.0
    except Exception as e:
        # 你也可以 print warning
        return 0.0


def _get_question_from_obj(obj: Dict[str, Any]) -> str:
    """
    兼容你说的格式：data["prompt"][0]["content"]
    也做一些保底。
    """
    p = obj.get("prompt", "")
    if isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
        return str(p[0].get("content", ""))
    if isinstance(p, str):
        return p
    return ""


def compute_score_router(
    obj: Dict[str, Any],
    solution_str: str,
    cfg: Optional[JudgeConfig] = None,
) -> float:
    """
    根据 data_source 分流：
    - LOCAL_SOURCES -> 走你现有 compute_score 逻辑
    - 其它 -> API judge
    """
    data_source: str = obj.get("data_source", "unknown")
    gt = obj["reward_model"]["ground_truth"]

    if data_source in LOCAL_SOURCES:
        return compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=gt,
        )

    # 走 API judge
    if cfg is None:
        # 没配置就保守返回 0
        return 0.0

    question = _get_question_from_obj(obj)
    return api_judge_score(question, gt, solution_str, cfg)


# =====================
# 5) Main eval
# =====================
def eval_pass_at_k(file_path: str, k: int, output_path: Optional[str] = None,
                   judge_cfg: Optional[JudgeConfig] = None) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    total = 0
    pass1_correct = 0.0
    passk_correct = 0.0

    for obj in _iter_records(file_path):
        responses = obj["responses"]
        if not responses:
            continue

        total += 1

        # pass@1
        pass1_correct += compute_score_router(
            obj=obj,
            solution_str=responses[0],
            cfg=judge_cfg,
        )

        # pass@k
        num_to_use = min(k, len(responses))
        any_correct_k = 0.0
        for sol in responses[:num_to_use]:
            score = compute_score_router(
                obj=obj,
                solution_str=sol,
                cfg=judge_cfg,
            )
            if score > 0:
                any_correct_k = 1.0
                break

        passk_correct += any_correct_k

    if total == 0:
        print("No valid samples found in file.")
        return

    pass1_acc = pass1_correct / total
    passk_acc = passk_correct / total

    print(f"File: {file_path}")
    print(f"Total samples: {total}")
    print(f"Pass@1 accuracy: {pass1_acc:.4%}")
    print(f"Pass@{k} accuracy (any of first {k} responses correct): {passk_acc:.4%}")

    # ===== 保存结果 =====
    if output_path is None:
        output_path = "recipe/GARO/evaluation_results.json"

    base_name = os.path.basename(file_path)
    file_stem, _ = os.path.splitext(base_name)

    metrics = {
        "file": file_path,
        "file_name": file_stem,
        "total_samples": total,
        "k": k,
        "pass_at_1": {
            "correct": pass1_correct,
            "accuracy": pass1_acc,
        },
        "pass_at_k": {
            "correct": passk_correct,
            "accuracy": passk_acc,
        },
    }

    if is_main_process():
        dir_path = os.path.dirname(output_path)
        local_mkdir(dir_path)

        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(metrics, fout, ensure_ascii=False, indent=2)

        print(f"Metrics written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to JSONL/Parquet with fields: prompt, responses, reward_model.ground_truth, data_source",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="k for pass@k. We look at the first k responses for each sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON",
    )

    # ===== Judge API args (optional) =====
    parser.add_argument("--judge_api_base", type=str, default=os.getenv("JUDGE_API_BASE", ""))
    parser.add_argument("--judge_api_key", type=str, default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge_model", type=str, default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge_timeout", type=int, default=int(os.getenv("JUDGE_TIMEOUT", "60")))
    parser.add_argument("--judge_temperature", type=float, default=float(os.getenv("JUDGE_TEMPERATURE", "0.0")))

    args = parser.parse_args()

    judge_cfg = None
    if args.judge_api_base and args.judge_api_key:
        judge_cfg = JudgeConfig(
            api_base=args.judge_api_base,
            api_key=args.judge_api_key,
            model=args.judge_model,
            timeout=args.judge_timeout,
            temperature=args.judge_temperature,
        )

    eval_pass_at_k(args.file, args.k, args.output, judge_cfg=judge_cfg)
