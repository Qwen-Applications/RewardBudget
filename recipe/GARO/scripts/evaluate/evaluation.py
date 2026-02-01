# -*- coding: utf-8 -*-
"""
GARO evaluation script (new, minimal, batch-judge via qwen_cluster style).

Goal:
  - Keep eval_all.sh / eval_all_16.sh calling working:
      python3 recipe/GARO/scripts/evaluate/evaluation.py --file ... --k ... --output ... --output_jsonl ...
  - Mainly support 4 args: file, k, output, output_jsonl
  - For LOCAL_SOURCES: use recipe.GARO.reward.compute_score
  - Otherwise: batch judge via vLLM OpenAI-completions-style endpoint (/v1/completions)
      (same request pattern as evaluate/qwen_cluster.py __main__ + send_request)

Judge prompt:
  - We assemble plain prompt string (qwen_cluster expects you to build prompt yourself)
  - Require output ONLY: \\boxed{correct} or \\boxed{incorrect}
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests
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
# 2) DDP helpers
# =====================
def is_main_process() -> bool:
    """只在未初始化 DDP 或 rank0 时返回 True。"""
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def local_mkdir(path: str) -> None:
    """只在 main process 创建目录，然后所有 rank 同步一下。"""
    if not path:
        return
    if is_main_process():
        os.makedirs(path, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# =====================
# 3) Input iterators (compatible with evaluate.zip's evaluation.py)
# =====================
def _iter_records(file_path: str) -> Iterator[Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".jsonl", ".json"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if ext == ".parquet":
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
            yield {k: v[i] for k, v in cols.items()}
        return

    raise ValueError(f"Unsupported file extension for {file_path}")


# =====================
# 4) Boxed extractor (brace-matching, consistent with recipe.GARO.reward)
# =====================
def extract_solution_inside_boxed(text: str) -> str:
    """Extract the LAST \\boxed{...} content, supporting nested braces."""
    if not text:
        return ""

    key = r"\boxed{"
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


def _get_question_from_obj(obj: Dict[str, Any]) -> str:
    """Support prompt being str or list[dict] with {content: ...}"""
    p = obj.get("prompt", "")
    if isinstance(p, list) and p and isinstance(p[0], dict):
        return str(p[0].get("content", ""))
    if isinstance(p, str):
        return p
    return ""


# =====================
# 5) Judge prompt / parse
# =====================

def apply_chat_template(messages: List[Dict[str, str]]) -> str:
    """Apply Qwen3 Instruct chat template for /v1/completions.

    Qwen3 Instruct models on Hugging Face use the ChatML-style special tokens:
      <|im_start|>{role}\n{content}<|im_end|>\n
    and for generation you append:
      <|im_start|>assistant\n
    This mirrors the tokenizer_config.json `chat_template` used by Qwen3-*-Instruct-2507.
    """
    parts: List[str] = []

    # Pull out an explicit system message if the first message is system.
    start_idx = 0
    if messages and (messages[0].get("role") or "").strip().lower() == "system":
        sys_content = messages[0].get("content") or ""
        parts.append(f"<|im_start|>system\n{sys_content}<|im_end|>\n")
        start_idx = 1

    for m in messages[start_idx:]:
        role = (m.get("role") or "user").strip().lower()
        content = m.get("content") or ""
        if role not in ("user", "assistant", "system"):
            role = "user"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    # add_generation_prompt
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_judge_prompt(question: str, ground_truth: str, pred_answer: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert grader for academic math benchmarks (e.g., AMC/AIME/Math500/OlympiadBench). "
                "Given a Problem, a Ground-truth Answer, and a model's Predicted Answer, decide if the predicted answer is correct.\n\n"
                "Rules:\n"
                "1) Focus ONLY on the final answer, not the solution steps.\n"
                "2) Ignore formatting differences (LaTeX, whitespace, punctuation, order) and accept mathematically equivalent forms.\n"
                "3) Enforce required domain/units/type (integer vs fraction, exact value vs approximation).\n"
                "4) If the problem expects an exact value, do NOT accept decimal approximations unless they are explicitly allowed.\n"
                "5) If a numeric approximation is acceptable, allow small rounding error (typical tolerance 1e-4 relative or absolute) unless the problem states otherwise.\n"
                "6) If multiple answers are valid (e.g., set/tuple), treat them as correct if they match as a set.\n\n"
                "Output format (strict): reply with exactly one term in the form \\boxed{correct} or \\boxed{incorrect}. "
                "Do not include any other words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\n"
                f"Ground-truth Answer:\n{ground_truth}\n\n"
                f"Predicted Answer:\n{pred_answer}\n"
                "Is the Predicted Answer correct? Reply with either \\boxed{correct} or \\boxed{incorrect}."
            ),
        },
    ]
    return apply_chat_template(messages)


def parse_judge_boxed(output_text: str) -> Optional[bool]:
    """Return True/False, or None if cannot parse."""
    if not output_text:
        return None
    boxed = extract_solution_inside_boxed(output_text)
    if not boxed:
        t = output_text.strip().lower()
        if t == "correct":
            return True
        if t == "incorrect":
            return False
        return None
    tag = boxed.strip().lower()
    if tag == "correct":
        return True
    if tag == "incorrect":
        return False
    return None


def _trunc(s: str, n: int = 400) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else (s[:n] + f"...(+{len(s)-n} chars)")


# =====================
# 6) qwen_cluster-style batch calling (NO dependency on more_itertools/tqdm)
#     Mirrors evaluate/qwen_cluster.py send_request payload format.
# =====================
@dataclass
class JudgeRuntime:
    ip: str
    port: str
    url_tmpl: str
    max_workers: int
    partition_size: int
    max_tokens: int
    timeout_seconds: int


def _get_judge_runtime(
    ip: str,
    port: str,
    url_tmpl: str,
    max_workers: int,
    partition_size: int,
    max_tokens: int,
    timeout_seconds: int,
) -> JudgeRuntime:
    return JudgeRuntime(
        ip=ip,
        port=port,
        url_tmpl=url_tmpl,
        max_workers=max_workers,
        partition_size=partition_size,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


def _resolve_url(rt: JudgeRuntime) -> str:
    url = rt.url_tmpl.replace("localhost", str(rt.ip)).replace("port", str(rt.port))
    return url


def _send_request(prompt: str, rt: JudgeRuntime, param_gen: Dict[str, Any]) -> str:
    """
    Exactly like evaluate/qwen_cluster.py: POST {url} with json {"prompt": prompt, **param_gen}
    and read result['choices'][-1]['text'].
    """
    url = _resolve_url(rt)
    payload = {"prompt": prompt, **param_gen}
    try:
        resp = requests.post(url, json=payload, timeout=rt.timeout_seconds)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][-1]["text"]
    except Exception as e:
        # keep behavior similar: print and return empty
        print(f"[JUDGE_HTTP_ERROR] {repr(e)} url={url}")
        return ""


def _chunked_indices(n: int, chunk_size: int) -> Iterator[range]:
    for start in range(0, n, chunk_size):
        yield range(start, min(n, start + chunk_size))


def call_judge_api_batch(prompts: List[str], rt: JudgeRuntime) -> List[str]:
    """Batch call with concurrency; preserves input order."""
    if not prompts:
        return []

    # param_gen matches qwen_cluster main style
    param_gen = {
        "max_tokens": rt.max_tokens,
        # keep deterministic for judge
        "temperature": 0.0,
    }

    outs: List[Optional[str]] = [None] * len(prompts)

    # Avoid multi-rank double billing
    if torch.distributed.is_initialized() and not is_main_process():
        return [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=rt.max_workers) as ex:
        for idx_range in _chunked_indices(len(prompts), rt.partition_size):
            future_to_idx = {
                ex.submit(_send_request, prompts[i], rt, param_gen): i for i in idx_range
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    outs[i] = fut.result()
                except Exception as e:
                    print(f"[JUDGE_FUTURE_ERROR] {repr(e)} idx={i}")
                    outs[i] = ""

    return [x or "" for x in outs]


# =====================
# 7) Verdict cache (question+gt+pred -> score)
# =====================
_JUDGE_VERDICT_CACHE: Dict[str, float] = {}
_JUDGE_PARSEFAIL_KEYS: set[str] = set()


def _cache_key(question: str, ground_truth: str, pred_answer: str) -> str:
    h = hashlib.sha1()
    h.update(question.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(ground_truth.encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(pred_answer.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def judge_scores_for_answers(question: str, ground_truth: str, pred_answers: List[str], rt: JudgeRuntime) -> List[float]:
    """
    pred_answers: list of extracted boxed answers ("" means empty)
    returns: list[float] of 1.0/0.0
    """
    # build uncached prompts
    uncached_idxs: List[int] = []
    uncached_prompts: List[str] = []

    for i, ans in enumerate(pred_answers):
        if not ans:
            continue
        ck = _cache_key(question, ground_truth, ans)
        if ck in _JUDGE_VERDICT_CACHE:
            continue
        uncached_idxs.append(i)
        uncached_prompts.append(build_judge_prompt(question, ground_truth, ans))

    outs: List[str] = []
    if uncached_prompts:
        outs = call_judge_api_batch(uncached_prompts, rt)

    # update cache
    for j, i in enumerate(uncached_idxs):
        ans = pred_answers[i]
        ck = _cache_key(question, ground_truth, ans)
        out = outs[j] if j < len(outs) else ""
        verdict = parse_judge_boxed(out)
        if verdict is None:
            print(
                "[JUDGE_ERROR] parse_failed "
                f"cache_key={ck} "
                f"question={_trunc(question)} "
                f"gt={_trunc(ground_truth)} "
                f"pred={_trunc(ans)} "
                f"judge_out={_trunc(out, 800)}"
            )
            _JUDGE_VERDICT_CACHE[ck] = 0.0
            _JUDGE_PARSEFAIL_KEYS.add(ck)
        else:
            _JUDGE_VERDICT_CACHE[ck] = 1.0 if verdict else 0.0
            if ck in _JUDGE_PARSEFAIL_KEYS:
                _JUDGE_PARSEFAIL_KEYS.remove(ck)

    # read scores
    scores: List[float] = []
    for ans in pred_answers:
        if not ans:
            scores.append(0.0)
            continue
        ck = _cache_key(question, ground_truth, ans)
        scores.append(_JUDGE_VERDICT_CACHE.get(ck, 0.0))
    return scores


# =====================
# 8) Main evaluation: pass@1, pass@k, optional per-response jsonl
# =====================
def eval_pass_at_k(file_path: str, k: int, output_path: Optional[str], output_jsonl: Optional[str], rt: JudgeRuntime, log_every: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    total = 0
    pass1_correct = 0.0
    passk_correct = 0.0
    llm_judge_fail = 0

    jsonl_f = None
    if output_jsonl and is_main_process():
        local_mkdir(os.path.dirname(output_jsonl))
        jsonl_f = open(output_jsonl, "w", encoding="utf-8")

    start_t = time.time()
    last_log_t = start_t
    log_every = max(int(log_every), 1)

    for obj in _iter_records(file_path):
        responses = obj.get("responses", None)
        if not responses:
            continue

        total += 1
        data_source = obj.get("data_source", "unknown")
        gt = obj["reward_model"]["ground_truth"]

        if is_main_process() and (total % log_every == 0):
            now = time.time()
            elapsed = now - start_t
            delta = now - last_log_t
            last_log_t = now
            pass1_sofar = (pass1_correct / total) if total > 0 else 0.0
            passk_sofar = (passk_correct / total) if total > 0 else 0.0
            speed = (log_every / delta) if delta > 0 else 0.0
            print(
                f"[PROGRESS] processed={total} "
                f"pass@1_sofar={pass1_sofar:.4%} "
                f"pass@{k}_sofar={passk_sofar:.4%} "
                f"llm_judge_fail={llm_judge_fail} "
                f"speed={speed:.2f} samples/s "
                f"elapsed={elapsed/60:.1f} min"
            )

        # need full per-response correctness
        if jsonl_f is not None:
            if data_source in LOCAL_SOURCES:
                scores = [compute_score(data_source=data_source, solution_str=sol, ground_truth=gt) for sol in responses]
            else:
                question = _get_question_from_obj(obj)
                pred_answers = [extract_solution_inside_boxed(sol) for sol in responses]
                scores = judge_scores_for_answers(question, gt, pred_answers, rt)
                for ans in pred_answers:
                    if not ans:
                        continue
                    ck = _cache_key(question, gt, ans)
                    if ck in _JUDGE_PARSEFAIL_KEYS:
                        llm_judge_fail += 1

            corrects = [1 if s > 0 else 0 for s in scores]
            pass1_correct += float(corrects[0])
            num_to_use = min(k, len(corrects))
            passk_correct += 1.0 if any(corrects[:num_to_use]) else 0.0

            jsonl_f.write(
                json.dumps(
                    {
                        "prompt": obj.get("prompt", ""),
                        "responses": responses,
                        "response_correct": corrects,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            continue

        # no output_jsonl: compute pass@1 & pass@k (batch judge for non-local)
        if data_source in LOCAL_SOURCES:
            s1 = compute_score(data_source=data_source, solution_str=responses[0], ground_truth=gt)
            pass1_correct += 1.0 if s1 > 0 else 0.0

            any_ok = 0.0
            for sol in responses[: min(k, len(responses))]:
                if compute_score(data_source=data_source, solution_str=sol, ground_truth=gt) > 0:
                    any_ok = 1.0
                    break
            passk_correct += any_ok
        else:
            question = _get_question_from_obj(obj)
            num_to_use = min(k, len(responses))
            pred_answers_k = [extract_solution_inside_boxed(sol) for sol in responses[:num_to_use]]
            scores_k = judge_scores_for_answers(question, gt, pred_answers_k, rt)

            for ans in pred_answers_k:
                if not ans:
                    continue
                ck = _cache_key(question, gt, ans)
                if ck in _JUDGE_PARSEFAIL_KEYS:
                    llm_judge_fail += 1

            pass1_correct += 1.0 if (scores_k[0] > 0) else 0.0
            passk_correct += 1.0 if any(s > 0 for s in scores_k) else 0.0

    if jsonl_f is not None:
        jsonl_f.close()
        print(f"Per-sample results written to: {output_jsonl}")

    if total == 0:
        print("No valid samples found in file.")
        return

    if is_main_process():
        elapsed = time.time() - start_t
        print(f"[PROGRESS] done processed={total} elapsed={elapsed/60:.1f} min")

    pass1_acc = pass1_correct / total
    passk_acc = passk_correct / total

    print(f"File: {file_path}")
    print(f"Total samples: {total}")
    print(f"Pass@1 accuracy: {pass1_acc:.4%}")
    print(f"Pass@{k} accuracy (any of first {k} responses correct): {passk_acc:.4%}")

    if output_path is None:
        output_path = "recipe/GARO/evaluation_results.json"

    if is_main_process():
        local_mkdir(os.path.dirname(output_path))
        base_name = os.path.basename(file_path)
        file_stem, _ = os.path.splitext(base_name)

        metrics = {
            "file": file_path,
            "file_name": file_stem,
            "total_samples": total,
            "k": k,
            "pass_at_1": {"correct": pass1_correct, "accuracy": pass1_acc},
            "pass_at_k": {"correct": passk_correct, "accuracy": passk_acc},
        }
        if llm_judge_fail > 0:
            metrics["llm_judge_fail"] = llm_judge_fail
        print("[METRICS]")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(metrics, fout, ensure_ascii=False, indent=2)

        print(f"Metrics written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=100, help="Print progress every N samples")
    # ---- qwen_cluster judge runtime (formerly env vars) ----
    parser.add_argument("--qwen_ip", type=str, default="10.2.4.24")
    parser.add_argument("--qwen_port", type=str, default="8000")
    parser.add_argument("--qwen_text_model_url", type=str, default="http://localhost:port/v1/completions")
    parser.add_argument("--qwen_max_workers", type=int, default=96)
    parser.add_argument("--qwen_partition_size", type=int, default=192)
    parser.add_argument("--qwen_max_tokens", type=int, default=800)
    parser.add_argument("--judge_timeout", type=int, default=300)
    args = parser.parse_args()

    rt = _get_judge_runtime(
        ip=args.qwen_ip,
        port=args.qwen_port,
        url_tmpl=args.qwen_text_model_url,
        max_workers=args.qwen_max_workers,
        partition_size=args.qwen_partition_size,
        max_tokens=args.qwen_max_tokens,
        timeout_seconds=args.judge_timeout,
    )

    eval_pass_at_k(args.file, args.k, args.output, args.output_jsonl, rt, args.log_every)


if __name__ == "__main__":
    main()
