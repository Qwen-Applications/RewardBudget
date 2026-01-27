#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference on one or more datasets and report pass@k.

This script is designed for "bring-your-own-checkpoint" evaluation:

  - ckpt: local HuggingFace model/ckpt path
  - datasets: comma-separated dataset names (e.g. math500,amc23,...)
  - k: number of generations per prompt (pass@k)

It writes:
  1) <output_dir>/<dataset>.generations.jsonl  (all generations + grading)
  2) <output_dir>/passk_summary.json           (pass@k per dataset)

Dataset resolution follows a pragmatic heuristic:

  1) Built-in: "math500" -> llm_experiments/data/MATH500.json
  2) If a dataset name is already a file path, use it directly.
  3) Otherwise, try <data_root>/GARO/test_data/<name>.{parquet,jsonl,json}

"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Optional deps
# -----------------------------

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None

try:
    from vllm import LLM, SamplingParams  # type: ignore

    _HAS_VLLM = True
except Exception:  # pragma: no cover
    _HAS_VLLM = False

try:
    import torch
    from transformers import AutoTokenizer  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer


# -----------------------------
# Utils
# -----------------------------


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic where possible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _normalize_dataset_name(name: str) -> str:
    return name.strip()


def _maybe_parse_json(x: Any) -> Any:
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x


def _extract_ground_truth(ex: Dict[str, Any]) -> Optional[str]:
    # Common patterns in the user's GARO parquet/jsonl:
    #   reward_model: {"ground_truth": "..."}
    rm = ex.get("reward_model")
    rm = _maybe_parse_json(rm)
    if isinstance(rm, dict):
        gt = rm.get("ground_truth") or rm.get("answer") or rm.get("gt")
        if gt is not None:
            return str(gt)

    # Common patterns in this repo's json:
    for k in ("answer", "ground_truth", "gt", "solution"):
        if k in ex and ex[k] is not None:
            return str(ex[k])

    return None


def _prompt_to_text(prompt: Any, tokenizer: Optional[Any]) -> str:
    if isinstance(prompt, str):
        return prompt

    # Some datasets store prompts as a list of {role, content} messages.
    if isinstance(prompt, list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            # Try best-effort: if elements are dicts with role/content.
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                try:
                    return tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass

        # Fallback join.
        try:
            return "\n".join(
                m.get("content", str(m)) if isinstance(m, dict) else str(m) for m in prompt
            )
        except Exception:
            return str(prompt)

    return str(prompt)


def _extract_final_answer(text: str) -> str:
    """Extract a short answer string from a model response."""

    # Prefer the last \boxed{...} / \fbox{...}
    boxed = parse_answer(text)
    if boxed is not None:
        return boxed.strip()

    # Heuristics: last non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]

    # A small cleanup for common "Answer: ..." patterns.
    for prefix in ("answer:", "final answer:", "final:"):
        if last.lower().startswith(prefix):
            return last[len(prefix) :].strip()

    return last


def _safe_grade(ans: str, gt: str) -> int:
    try:
        return int(grade_answer(ans, gt))
    except Exception:
        return 0


# -----------------------------
# Dataset IO
# -----------------------------


@dataclass
class DatasetSpec:
    name: str
    path: Path


def resolve_datasets(
    dataset_arg: str,
    data_root: Optional[str],
    repo_root: Path,
) -> List[DatasetSpec]:
    """Resolve a comma-separated dataset argument into local file paths."""

    names = [_normalize_dataset_name(x) for x in dataset_arg.split(",") if x.strip()]
    specs: List[DatasetSpec] = []

    builtins = {
        "math500": repo_root / "llm_experiments" / "data" / "MATH500.json",
        "MATH500": repo_root / "llm_experiments" / "data" / "MATH500.json",
    }

    for name in names:
        # If user passes an explicit file path, use it.
        p = Path(name)
        if p.exists() and p.is_file():
            specs.append(DatasetSpec(name=p.stem, path=p))
            continue

        if name in builtins and builtins[name].exists():
            specs.append(DatasetSpec(name=name.lower(), path=builtins[name]))
            continue

        # Try GARO-style layout.
        if data_root is None:
            raise FileNotFoundError(
                f"Cannot resolve dataset '{name}'. Provide --data_root, or pass a file path."
            )

        cand_dir = Path(data_root) / "GARO" / "test_data"
        cands = [
            cand_dir / f"{name}.parquet",
            cand_dir / f"{name}.jsonl",
            cand_dir / f"{name}.json",
        ]
        found = next((c for c in cands if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Cannot find dataset file for '{name}'. Tried: {', '.join(map(str, cands))}"
            )
        specs.append(DatasetSpec(name=name, path=found))

    return specs


def iter_examples(path: Path, batch_size: int = 256) -> Iterator[Dict[str, Any]]:
    """Yield raw example dicts from parquet/json/jsonl."""

    suf = path.suffix.lower()
    if suf == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for ex in data:
                yield ex
        elif isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for ex in data["data"]:
                yield ex
        else:
            raise ValueError(f"Unsupported JSON format at {path}")
        return

    if suf == ".parquet":
        if pq is None:
            raise ImportError("pyarrow is required to read parquet datasets")
        pf = pq.ParquetFile(path)
        # Read in batches to avoid loading the entire dataset.
        for rb in pf.iter_batches(batch_size=batch_size):
            for ex in rb.to_pylist():
                yield ex
        return

    raise ValueError(f"Unsupported dataset file type: {path}")


# -----------------------------
# Generation backends
# -----------------------------


@dataclass
class GenConfig:
    k: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int


class Generator:
    def generate_many(self, prompts: Sequence[str], cfg: GenConfig) -> List[List[str]]:
        raise NotImplementedError


class VLLMGenerator(Generator):
    def __init__(
        self,
        ckpt: str,
        trust_remote_code: bool,
        tensor_parallel_size: int,
        dtype: str,
        max_model_len: Optional[int],
    ) -> None:
        if not _HAS_VLLM:
            raise RuntimeError("vllm is not installed")
        kwargs: Dict[str, Any] = {
            "model": ckpt,
            "trust_remote_code": trust_remote_code,
            "tensor_parallel_size": tensor_parallel_size,
        }
        if dtype:
            kwargs["dtype"] = dtype
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        self.llm = LLM(**kwargs)

    def generate_many(self, prompts: Sequence[str], cfg: GenConfig) -> List[List[str]]:
        # vLLM: if temperature == 0, generation is greedy and deterministic.
        sp = SamplingParams(
            n=cfg.k,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        outs = self.llm.generate(list(prompts), sp)
        all_gens: List[List[str]] = []
        for o in outs:
            gens = [x.text for x in o.outputs]
            all_gens.append(gens)
        return all_gens


class HFGenerator(Generator):
    def __init__(
        self,
        ckpt: str,
        trust_remote_code: bool,
        dtype: str,
        device: str,
        max_prompt_tokens: Optional[int],
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch/transformers are not installed")

        from transformers import AutoModelForCausalLM  # local import

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt, trust_remote_code=trust_remote_code
        )
        torch_dtype = None
        if dtype:
            # e.g. "float16", "bfloat16"
            torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_prompt_tokens = max_prompt_tokens

    def generate_many(self, prompts: Sequence[str], cfg: GenConfig) -> List[List[str]]:
        do_sample = cfg.temperature > 0

        all_gens: List[List[str]] = []
        for i, p in enumerate(prompts):
            # Make sampling deterministic across prompts.
            torch.manual_seed(cfg.seed + i)
            toks = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=self.max_prompt_tokens is not None,
                max_length=self.max_prompt_tokens,
            ).to(self.device)

            gen_ids = self.model.generate(
                **toks,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p if do_sample else None,
                top_k=cfg.top_k if do_sample else None,
                num_return_sequences=cfg.k,
            )

            # Remove the prompt part.
            prompt_len = toks["input_ids"].shape[1]
            gens: List[str] = []
            for seq in gen_ids:
                gen_text = self.tokenizer.decode(
                    seq[prompt_len:], skip_special_tokens=True
                )
                gens.append(gen_text)
            all_gens.append(gens)

        return all_gens


# -----------------------------
# Main evaluation loop
# -----------------------------


def run_dataset(
    ds: DatasetSpec,
    gen: Generator,
    tokenizer: Optional[Any],
    cfg: GenConfig,
    out_dir: Path,
    batch_size: int,
    write_prompt_text: bool,
) -> Dict[str, Any]:
    out_jsonl = out_dir / f"{ds.name}.generations.jsonl"
    tmp_jsonl = out_dir / f"{ds.name}.generations.jsonl.tmp"

    total = 0
    passk_hits = 0

    # Stream writes to avoid holding the whole dataset in memory.
    tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_f = tmp_jsonl.open("w", encoding="utf-8")

    # Batch prompts for efficiency.
    prompt_buf: List[str] = []
    raw_buf: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal total, passk_hits, prompt_buf, raw_buf
        if not prompt_buf:
            return

        gens = gen.generate_many(prompt_buf, cfg)
        assert len(gens) == len(raw_buf)

        for ex, prompt_text, ex_gens in zip(raw_buf, prompt_buf, gens):
            gt = _extract_ground_truth(ex)
            if gt is None:
                # Skip ungradeable examples, but keep them in output.
                gt = ""

            gen_infos = []
            any_correct = False
            for g in ex_gens:
                ans = _extract_final_answer(g)
                correct = bool(_safe_grade(ans, gt)) if gt else False
                any_correct = any_correct or correct
                gen_infos.append({"text": g, "answer": ans, "correct": correct})

            rec: Dict[str, Any] = {
                "dataset": ds.name,
                "idx": total,
                "id": ex.get("id"),
                "ground_truth": gt,
                "prompt": ex.get("prompt"),
                "generations": gen_infos,
                "pass": any_correct,
            }
            if write_prompt_text:
                rec["prompt_text"] = prompt_text

            tmp_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += 1
            if any_correct:
                passk_hits += 1

        prompt_buf = []
        raw_buf = []

    for ex in iter_examples(ds.path, batch_size=batch_size):
        prompt = ex.get("prompt")
        prompt = _maybe_parse_json(prompt)
        prompt_text = _prompt_to_text(prompt, tokenizer)
        prompt_buf.append(prompt_text)
        raw_buf.append(ex)
        if len(prompt_buf) >= batch_size:
            flush()

    flush()

    tmp_f.close()
    tmp_jsonl.replace(out_jsonl)

    denom = max(total, 1)
    passk = passk_hits / denom

    metrics = {
        "dataset": ds.name,
        "path": str(ds.path),
        "num_examples": total,
        "k": cfg.k,
        "pass@k": passk,
        "output_jsonl": str(out_jsonl),
    }

    # Per-dataset metrics file (handy for shell pipelines).
    _json_dump(metrics, out_dir / f"{ds.name}.metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic inference and compute pass@k per dataset."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Local HF ckpt/model path")
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated dataset names (or file paths)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.environ.get("DATA_DIR") or os.environ.get("GARO_DATA_DIR"),
        help=(
            "Root directory for GARO-style datasets (expects GARO/test_data/<name>.parquet). "
            "Defaults to $DATA_DIR or $GARO_DATA_DIR if set."
        ),
    )
    parser.add_argument("-k", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inference",
        help="Output directory for jsonl + summary",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "hf"],
        default="vllm" if _HAS_VLLM else "hf",
        help="Generation backend (default: vllm if installed)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation (prompts per batch)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy/deterministic)",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for deterministic sampling)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF/vLLM",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="",
        help="dtype for vLLM/HF (e.g. float16, bfloat16). Empty = default",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM max_model_len (optional)",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="HF-only: truncate prompt to this many tokens (optional)",
    )
    parser.add_argument(
        "--write_prompt_text",
        action="store_true",
        help="Also store rendered prompt_text in jsonl (can be large)",
    )

    args = parser.parse_args()

    _set_global_determinism(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = resolve_datasets(args.datasets, args.data_root, repo_root)

    gen_cfg = GenConfig(
        k=int(args.k),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=int(args.seed),
    )

    # NOTE: vLLM doesn't need tokenizer for chat-template rendering, but we use it
    # to render list-of-messages prompts into a single string when possible.
    tokenizer = None
    if _HAS_TORCH:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.ckpt, trust_remote_code=args.trust_remote_code
            )
        except Exception:
            tokenizer = None

    if args.backend == "vllm":
        gen: Generator = VLLMGenerator(
            ckpt=args.ckpt,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=int(args.tensor_parallel_size),
            dtype=args.dtype,
            max_model_len=args.max_model_len,
        )
    else:
        gen = HFGenerator(
            ckpt=args.ckpt,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            device="cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu",
            max_prompt_tokens=args.max_prompt_tokens,
        )

    summary: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "ckpt": args.ckpt,
        "backend": args.backend,
        "sampling": {
            "k": gen_cfg.k,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "seed": gen_cfg.seed,
        },
        "datasets": {},
    }

    for ds in datasets:
        print(f"[INFO] Dataset={ds.name} path={ds.path}")
        metrics = run_dataset(
            ds=ds,
            gen=gen,
            tokenizer=tokenizer,
            cfg=gen_cfg,
            out_dir=out_dir,
            batch_size=int(args.batch_size),
            write_prompt_text=bool(args.write_prompt_text),
        )
        summary["datasets"][ds.name] = metrics
        print(
            f"[INFO] {ds.name}: pass@{gen_cfg.k}={metrics['pass@k']:.4f} "
            f"(n={metrics['num_examples']}) -> {metrics['output_jsonl']}"
        )

    _json_dump(summary, out_dir / "passk_summary.json")
    print(f"[INFO] Wrote summary: {out_dir / 'passk_summary.json'}")


if __name__ == "__main__":
    main()
