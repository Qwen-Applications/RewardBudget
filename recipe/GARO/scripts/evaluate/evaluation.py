# recipe/GARO/eval_pass_k.py
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple

import torch

from recipe.GARO.reward import compute_score

# =====================
# 1) data_source routing
# =====================
# 在 LOCAL_SOURCES 里面的，走 exact-match；其他的走 API
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
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "pyarrow is required to read parquet files. Please install pyarrow or convert to JSONL."
            ) from e

        table = pq.read_table(file_path)
        cols = table.to_pydict()
        n = table.num_rows

        for i in range(n):
            row = {k: v[i] for k, v in cols.items()}
            yield row
    else:
        raise ValueError(f"Unsupported file extension for {file_path}")


# =====================
# 3) Boxed extractor (brace-matching, consistent with recipe.GARO.reward)
# =====================
_BOXED_RE = re.compile(r"\\boxed\{")


def extract_solution_inside_boxed(text: str) -> str:
    """Extract the LAST \boxed{...} content.

    This brace-matching version supports nested braces (robust for LaTeX).
    Returns "" if no boxed is found.
    """
    if not text:
        return ""

    # The model outputs a single backslash in Python strings (i.e. "\\boxed{...")
    # so we search for r"\boxed{".
    key = r"\boxed{"  # one backslash in the *text*
    i, n = 0, len(text)
    last: Optional[str] = None

    while i < n:
        start = text.find(key, i)
        if start == -1:
            break
        j = start + len(key)
        depth = 1
        content_start = j
        while j < n and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[content_start : j - 1]
            i = j
        else:
            break
    return (last or "").strip()


# =====================
# 4) API Judge
# =====================


@dataclass
class JudgeConfig:
    """Judge backend configuration.

    Supported backends:
      - qwen_primus: use recipe/GARO/scripts/evaluate/qwen_primus.py (SSE)
      - openai_compat: OpenAI-compatible /v1/chat/completions endpoint
    """

    backend: str = "qwen_primus"

    # --- openai_compat ---
    api_base: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout: int = 60
    temperature: float = 0.0

    # --- qwen_primus ---
    qwen_api_key: str = ""
    qwen_text_model_url: str = ""
    qwen_model_name: str = "qwen3-235b-instruct-trans-sg"
    qwen_verbose: bool = False


def build_judge_prompt(question: str, ground_truth: str, pred_answer: str) -> str:
    return f"""Judge if the model final answer matches the ground truth for the given problem.

Be strict on meaning, not format: ignore LaTeX/whitespace/notation; accept mathematically equivalent forms.
Respect required domain/units/exactness; allow small rounding error.

Reply ONLY with \\boxed{{correct}} or \\boxed{{incorrect}}.

Problem: {question}
Ground truth: {ground_truth}
Model answer: {pred_answer}
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
    # Extract last boxed content if present
    boxed = extract_solution_inside_boxed(output_text)
    if not boxed:
        lowered = output_text.strip().lower()
        if lowered == "correct":
            return True
        if lowered == "incorrect":
            return False
        return None
    tag = boxed.strip().lower()
    if tag == "correct":
        return True
    if tag == "incorrect":
        return False
    return None


_QWEN_HANDLER_CACHE: Dict[Tuple[str, str, str, bool], Any] = {}


def _get_qwen_handler(cfg: JudgeConfig):
    from recipe.GARO.scripts.evaluate.qwen_primus import LLMHandler

    # allow empty fields -> fall back to defaults inside LLMHandler
    api_key = cfg.qwen_api_key or os.getenv("QWEN_API_KEY", "")
    url = cfg.qwen_text_model_url or os.getenv("QWEN_TEXT_MODEL_URL", "")
    model_name = cfg.qwen_model_name or os.getenv("QWEN_MODEL_NAME", "qwen3-235b-instruct-trans-sg")
    verbose = bool(cfg.qwen_verbose)

    cache_key = (api_key, url, model_name, verbose)
    if cache_key not in _QWEN_HANDLER_CACHE:
        _QWEN_HANDLER_CACHE[cache_key] = LLMHandler(
            text_model_name=model_name,
            api_key=api_key or None,
            text_model_url=url or None,
            verbose=verbose,
            timeout=cfg.timeout,
        )
    return _QWEN_HANDLER_CACHE[cache_key]


def call_judge_api(prompt: str, cfg: JudgeConfig) -> str:
    """Call judge model with the configured backend."""

    if cfg.backend == "qwen_primus":
        handler = _get_qwen_handler(cfg)
        messages = [
            {"role": "system", "content": "You are a precise grading assistant."},
            {"role": "user", "content": prompt},
        ]
        # qwen_primus returns a plain string in non-stream mode
        return handler.call_qwen3(messages, stream=False, model_name=cfg.qwen_model_name)

    if cfg.backend == "openai_compat":
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
        return data["choices"][0]["message"]["content"]

    raise ValueError(f"Unknown judge backend: {cfg.backend}")


_JUDGE_VERDICT_CACHE: Dict[str, float] = {}


def _cache_key(question: str, ground_truth: str, pred_answer: str) -> str:
    h = hashlib.sha1()
    h.update(question.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(ground_truth.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(pred_answer.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def api_judge_score(question: str, ground_truth: str, solution_str: str, cfg: JudgeConfig) -> float:
    """
    使用 API judge 判断 correctness。
    返回 1.0/0.0
    """
    pred_answer = extract_solution_inside_boxed(solution_str)
    prompt = build_judge_prompt(question, ground_truth, pred_answer)

    ck = _cache_key(question, ground_truth, pred_answer)
    if ck in _JUDGE_VERDICT_CACHE:
        return _JUDGE_VERDICT_CACHE[ck]

    # 重要：如果你真的在 DDP 环境跑评测，避免多 rank 重复扣费
    if torch.distributed.is_initialized() and not is_main_process():
        return 0.0

    try:
        out = call_judge_api(prompt, cfg)
        verdict = parse_judge_boxed(out)
        score = 1.0 if verdict else 0.0
        _JUDGE_VERDICT_CACHE[ck] = score
        return score
    except Exception as e:
        print(f"[WARN] judge failed: {type(e).__name__}: {e}.")
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

    # ===== Judge backend args =====
    parser.add_argument(
        "--judge_backend",
        type=str,
        default=os.getenv("JUDGE_BACKEND", "qwen_primus"),
        choices=["qwen_primus", "openai_compat"],
        help="LLM-as-judge backend. Default: qwen_primus",
    )

    # ---- openai_compat ----
    parser.add_argument("--judge_api_base", type=str, default=os.getenv("JUDGE_API_BASE", ""))
    parser.add_argument("--judge_api_key", type=str, default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge_model", type=str, default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge_timeout", type=int, default=int(os.getenv("JUDGE_TIMEOUT", "60")))
    parser.add_argument("--judge_temperature", type=float, default=float(os.getenv("JUDGE_TEMPERATURE", "0.0")))

    # ---- qwen_primus ----
    parser.add_argument("--qwen_api_key", type=str, default="REDACTED_QWEN_API_KEY")
    parser.add_argument("--qwen_text_model_url", type=str, default="http://shenma-ai-pub.alibaba-inc.com/v1/completions")
    parser.add_argument("--qwen_model_name", type=str, default="qwen3-235b-instruct-trans-sg")
    parser.add_argument("--qwen_verbose", action="store_true", help="Print qwen request_id / debug logs")

    args = parser.parse_args()

    # Build judge config.
    # - qwen_primus: enabled by default (uses defaults in qwen_primus.py if you don't pass key/url)
    # - openai_compat: only enabled if api_base/api_key are provided
    judge_cfg = None
    if args.judge_backend == "qwen_primus":
        judge_cfg = JudgeConfig(
            backend="qwen_primus",
            timeout=args.judge_timeout,
            qwen_api_key=args.qwen_api_key,
            qwen_text_model_url=args.qwen_text_model_url,
            qwen_model_name=args.qwen_model_name,
            qwen_verbose=args.qwen_verbose,
        )
    else:
        if args.judge_api_base and args.judge_api_key:
            judge_cfg = JudgeConfig(
                backend="openai_compat",
                api_base=args.judge_api_base,
                api_key=args.judge_api_key,
                model=args.judge_model,
                timeout=args.judge_timeout,
                temperature=args.judge_temperature,
            )

    eval_pass_at_k(args.file, args.k, args.output, judge_cfg=judge_cfg)
