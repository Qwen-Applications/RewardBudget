#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference on one or more datasets and report pass@k.

This script is designed for "bring-your-own-checkpoint" evaluation:

  - ckpt: local HuggingFace model/ckpt path
  - datasets: comma-separated dataset names (e.g. math500,amc23,...)
  - k: number of generations per prompt (pass@k)

It writes (legacy backends: vllm/hf):
  1) <output_dir>/<dataset>_powers.generations.jsonl  (all generations + grading)
  2) <output_dir>/passk_summary.json           (pass@k per dataset)

It writes (verl backend, matching recipe/GARO/scripts/evaluate/eval_all.sh):
  1) <output_dir>/GARO/<dataset>_<run_tag>.parquet           (responses in parquet)
  2) <output_dir>/GARO/<dataset>_<run_tag>.eval.json         (summary metrics)
  3) <output_dir>/GARO/<dataset>_<run_tag>.eval.details.jsonl (optional per-sample correctness)
  4) <output_dir>/GARO/passk_summary.json                    (aggregated metrics)

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
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Sequence,
                    Tuple, Union)

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


try:
    import importlib.util as _importlib_util

    _HAS_VERL = _importlib_util.find_spec("verl") is not None
except Exception:  # pragma: no cover
    _HAS_VERL = False


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

def _default_ckpts_from_env() -> List[str]:
    """
    Default to the same two model IDs commonly used in eval scripts:
      ${PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR} with format "qwen:path1;llama:path2"
    """
    multi_base = os.environ.get("PRIMUS_MULTI_SOURCE_CHECKPOINT_DIR")
    if not multi_base:
        return []
    
    ckpt_paths = []
    # Parse format: "qwen:path1;llama:path2"
    model_entries = multi_base.split(";")
    for entry in model_entries:
        if ":" in entry:
            model, path = entry.split(":", 1)
            base = path.strip()
            model = "Qwen3-8B" if model.lower() == "qwen" else "Llama-3.2-3B-Instruct"
            ckpt_paths.append(str(Path(base, model)))
        else:
            # If no colon, treat entire entry as path
            ckpt_paths.append(entry.strip())
    
    return ckpt_paths


def _split_csv(x: Optional[str]) -> List[str]:
    if not x:
        return []
    return [s.strip() for s in x.split(",") if s.strip()]


def _sanitize_tag(x: str) -> str:
    # Keep it filesystem-friendly.
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in x)


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
# eval_all.sh-compatible (verl)
# -----------------------------


def _set_eval_all_env_defaults() -> None:
    """Best-effort defaults mirroring recipe/GARO/scripts/evaluate/eval_all.sh exports."""

    # Only set defaults if user didn't set them.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    os.environ.setdefault("VLLM_USE_NCCL_SYMM_MEM", "0")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")


def _sanitize_tag(s: str) -> str:
    s = s.strip()
    if not s:
        return "run"
    # Keep it filename-safe.
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "run"


def _default_run_tag(ckpt: str) -> str:
    base = Path(ckpt).name or "ckpt"
    return _sanitize_tag(base)


def _hydra_list_str(items: Sequence[Union[str, Path]]) -> str:
    # Hydra list syntax, safe quoting via json.
    return "[" + ",".join(json.dumps(str(x)) for x in items) + "]"


def _run_cmd(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
    joined = " ".join(cmd)
    print(f"[CMD] (cwd={cwd}) {joined}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _run_verl_generation_multi(
    *,
    project_root: Path,
    ckpt: str,
    data_paths: Sequence[Path],
    output_paths: Sequence[Path],
    nnodes: int,
    n_gpus_per_node: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    prompt_key: str,
    prompt_length: int,
    response_length: int,
    trust_remote_code: bool,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    actor_rollout_mode: str,
    verl_batch_size: int,
) -> None:
    """Run verl.trainer.main_generation_multi similarly to eval_all.sh."""

    if temperature == 0.0 and n_samples != 1:
        raise ValueError(
            "verl.trainer.main_generation_multi asserts: when temperature=0, n_samples must be 1. "
            "Set --temperature > 0 to sample multiple responses (k>1)."
        )

    # Ensure parent directories exist (verl will create as needed, but being explicit helps).
    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_generation_multi",
        f"trainer.nnodes={int(nnodes)}",
        f"trainer.n_gpus_per_node={int(n_gpus_per_node)}",
        f"+data.paths={_hydra_list_str(data_paths)}",
        f"+data.output_paths={_hydra_list_str(output_paths)}",
        f"data.prompt_key={prompt_key}",
        f"data.n_samples={int(n_samples)}",
        f"data.batch_size={int(verl_batch_size)}",
        f"model.path={ckpt}",
        # main_generation_multi currently reads trust_remote_code from config.data (not config.model).
        # We set both to be robust.
        f"+data.trust_remote_code={str(bool(trust_remote_code))}",
        f"+model.trust_remote_code={str(bool(trust_remote_code))}",
        f"rollout.temperature={float(temperature)}",
        f"rollout.top_p={float(top_p)}",
        f"rollout.prompt_length={int(prompt_length)}",
        f"rollout.response_length={int(response_length)}",
        f"rollout.tensor_model_parallel_size={int(tensor_model_parallel_size)}",
        f"+rollout.pipeline_model_parallel_size={int(pipeline_model_parallel_size)}",
        f"+actor_rollout_ref.rollout.mode={actor_rollout_mode}",
    ]

    # Mirror eval_all.sh environment knobs (best-effort).
    _set_eval_all_env_defaults()
    _run_cmd(cmd, cwd=project_root)


def _run_recipe_evaluation(
    *,
    project_root: Path,
    file_path: Path,
    k: int,
    output_json: Path,
    output_jsonl: Optional[Path],
    judge_backend: Optional[str],
) -> None:
    eval_script = project_root / "recipe" / "GARO" / "scripts" / "evaluate" / "evaluation.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Cannot find evaluation.py at: {eval_script}")

    cmd = [
        sys.executable,
        str(eval_script),
        "--file",
        str(file_path),
        "--k",
        str(int(k)),
        "--output",
        str(output_json),
    ]
    if output_jsonl is not None:
        cmd += ["--output_jsonl", str(output_jsonl)]
    if judge_backend:
        cmd += ["--judge_backend", judge_backend]

    _run_cmd(cmd, cwd=project_root)


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

        # Prefer GARO-style layout if data_root is provided (common in prod eval).
        tried: List[Path] = []
        if data_root is not None:
            cand_dir = Path(data_root) / "GARO" / "test_data"
            cands = [
                cand_dir / f"{name}.parquet",
                cand_dir / f"{name}.jsonl",
                cand_dir / f"{name}.json",
            ]
            tried.extend(cands)
            found = next((c for c in cands if c.exists()), None)
            if found is not None:
                specs.append(DatasetSpec(name=name, path=found))
                continue

        # Fall back to built-ins (repo-provided MATH500.json).
        if name in builtins and builtins[name].exists():
            specs.append(DatasetSpec(name=name.lower(), path=builtins[name]))
            continue

        # Try GARO-style layout.
        if data_root is None:
            raise FileNotFoundError(
                f"Cannot resolve dataset '{name}'. Provide --data_root, or pass a file path."
            )

        raise FileNotFoundError(
            f"Cannot find dataset file for '{name}'. Tried: {', '.join(map(str, tried))}"
        )

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
    ds_tag = f"{ds.name}_powers"
    out_jsonl = out_dir / f"{ds_tag}.generations.jsonl"
    tmp_jsonl = out_dir / f"{ds_tag}.generations.jsonl.tmp"

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
                "dataset": ds_tag,
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
        "dataset": ds_tag,
        "path": str(ds.path),
        "num_examples": total,
        "k": cfg.k,
        "pass@k": passk,
        "output_jsonl": str(out_jsonl),
    }

    # Per-dataset metrics file (handy for shell pipelines).
    _json_dump(metrics, out_dir / f"{ds_tag}.metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic inference and compute pass@k per dataset."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Local HF ckpt/model path. If omitted (and --ckpts is also omitted), "
            "defaults to two models under $PRIMUS_SOURCE_CHECKPOINT_DIR: "
            "Qwen3-8B and Llama-3.2-3B-Instruct."
        ),
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default=None,
        help="Comma-separated list of local HF ckpt/model paths (overrides --ckpt).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="math500,amc23,aime24,aime25,omni_math,minerva,OlympiadBench_math_en_comp,OlympiadBench_physics_en_comp",
        help="Comma-separated dataset names (or file paths)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/primus_datasets/zmy",
        help=(
            "Root directory for GARO-style datasets (expects GARO/test_data/<name>.parquet). "
            "Defaults to $DATA_DIR or $GARO_DATA_DIR if set."
        ),
    )
    parser.add_argument("-k", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expandvars("${OUTPUT_DIR}/GARO/"),
        help=(
            "Output directory (parent). For verl backend, outputs are written under <output_dir>/GARO. "
            "For legacy backends, jsonl + summary are written directly under <output_dir>."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["verl", "vllm", "hf"],
        default="verl" if (Path(__file__).resolve().parents[2] / "verl").exists() else ("vllm" if _HAS_VLLM else "hf"),
        help=(
            "Generation backend. 'verl' matches recipe/GARO/scripts/evaluate/eval_all.sh "
            "(calls verl.trainer.main_generation_multi and recipe evaluation)."
        ),
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
        default=8192,
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
        help="Tensor parallel size (vLLM) / rollout.tensor_model_parallel_size (verl)",
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

    # ---- verl (eval_all.sh-compatible) knobs ----
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Tag used in output filenames for verl backend. If empty, derived from --model/--step or --ckpt basename."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model name used to build default run_tag (verl backend only).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="",
        help="Optional step used to build default run_tag (verl backend only).",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/root/tmp",
        help="Temporary directory for intermediate parquet outputs (verl backend).",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="Parquet column name containing chat prompts (verl backend).",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="trainer.nnodes for verl generation (default: $NNODES or 1).",
    )
    parser.add_argument(
        "--n_gpus_per_node",
        type=int,
        default=1,
        help="trainer.n_gpus_per_node for verl generation (default: $NUM_ACCELERATORS or 1).",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=1024,
        help="rollout.prompt_length for verl generation (default: 1024).",
    )
    parser.add_argument(
        "--response_length",
        type=int,
        default=8192,
        help="rollout.response_length for verl generation (default: 8192).",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="+rollout.pipeline_model_parallel_size for verl generation (default: 1).",
    )
    parser.add_argument(
        "--actor_rollout_mode",
        type=str,
        default="async",
        choices=["async", "sync"],
        help="+actor_rollout_ref.rollout.mode for verl generation (default: async).",
    )
    parser.add_argument(
        "--verl_batch_size",
        type=int,
        default=128,
        help="data.batch_size for verl generation (default: 128).",
    )
    parser.add_argument(
        "--judge_backend",
        type=str,
        default='qwen_primus',
        choices=["qwen_primus", "openai_compat"],
        help=(
            "Judge backend passed to recipe/GARO/scripts/evaluate/evaluation.py (verl backend). "
            "If unset, evaluation.py will use its own defaults/env vars."
        ),
    )
    parser.add_argument(
        "--no_eval_details",
        action="store_true",
        help="Do not write per-sample correctness jsonl (verl backend).",
    )

    args = parser.parse_args()

    _set_global_determinism(args.seed)

    powers_root = Path(__file__).resolve().parents[1]
    project_root = Path(__file__).resolve().parents[2]

    datasets = resolve_datasets(args.datasets, args.data_root, powers_root)

    # ---------------------
    # verl backend
    # ---------------------
    if args.backend == "verl":
        if not (project_root / "verl").exists():
            raise RuntimeError(
                "Backend 'verl' selected but 'verl/' package not found under project root. "
                "Run from the repo root or use --backend vllm/hf."
            )

        # verl expects parquet input (as in eval_all.sh)
        non_parquet = [ds for ds in datasets if ds.path.suffix.lower() != ".parquet"]
        if non_parquet:
            bad = ", ".join(f"{ds.name}={ds.path}" for ds in non_parquet)
            raise ValueError(
                "verl backend only supports parquet inputs (same as eval_all.sh). "
                f"Got non-parquet dataset(s): {bad}"
            )

        out_parent = Path(args.output_dir)
        out_dir = out_parent if out_parent.name == "GARO" else (out_parent / "GARO")
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = Path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        run_tag = (args.run_tag or "").strip()
        if not run_tag:
            if (args.model or "").strip() and (args.step or "").strip():
                run_tag = f"{args.model.strip()}_{args.step.strip()}"
            elif (args.model or "").strip():
                run_tag = args.model.strip()
            else:
                run_tag = _default_run_tag(args.ckpt)
        run_tag = _sanitize_tag(run_tag)

        data_paths = [ds.path for ds in datasets]
        tmp_paths = [tmp_dir / f"{ds.name}_{run_tag}.parquet" for ds in datasets]

        eval_paths = [out_dir / f"{ds.name}_{run_tag}.eval.json" for ds in datasets]
        eval_jsonl_paths = [
            (out_dir / f"{ds.name}_{run_tag}.eval.details.jsonl") for ds in datasets
        ]

        print(
            f"[INFO] Backend=verl | run_tag={run_tag} | out_dir={out_dir} | tmp_dir={tmp_dir}"
        )
        print(f"[INFO] Checkpoint path: {args.ckpt}")
        print(f"[INFO] Data path(s): {' '.join(map(str, data_paths))}")

        _run_verl_generation_multi(
            project_root=project_root,
            ckpt=args.ckpt,
            data_paths=data_paths,
            output_paths=tmp_paths,
            nnodes=int(args.nnodes),
            n_gpus_per_node=int(args.n_gpus_per_node),
            n_samples=int(args.k),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            prompt_key=str(args.prompt_key),
            prompt_length=int(args.prompt_length),
            response_length=int(args.response_length),
            trust_remote_code=bool(args.trust_remote_code),
            tensor_model_parallel_size=int(args.tensor_parallel_size),
            pipeline_model_parallel_size=int(args.pipeline_model_parallel_size),
            actor_rollout_mode=str(args.actor_rollout_mode),
            verl_batch_size=int(args.verl_batch_size),
        )

        summary: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "ckpt": args.ckpt,
            "backend": "verl",
            "run_tag": run_tag,
            "generation": {
                "k": int(args.k),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "prompt_length": int(args.prompt_length),
                "response_length": int(args.response_length),
                "nnodes": int(args.nnodes),
                "n_gpus_per_node": int(args.n_gpus_per_node),
                "tensor_model_parallel_size": int(args.tensor_parallel_size),
                "pipeline_model_parallel_size": int(args.pipeline_model_parallel_size),
                "actor_rollout_mode": str(args.actor_rollout_mode),
                "batch_size": int(args.verl_batch_size),
            },
            "datasets": {},
        }

        # Copy parquet into output dir (mirrors eval_all.sh's rclone copy) + evaluate
        for ds, tmp_p, eval_p, eval_j in zip(datasets, tmp_paths, eval_paths, eval_jsonl_paths):
            if not tmp_p.exists():
                raise FileNotFoundError(f"verl output parquet not found: {tmp_p}")

            out_parquet = out_dir / tmp_p.name
            shutil.copy2(tmp_p, out_parquet)

            _run_recipe_evaluation(
                project_root=project_root,
                file_path=tmp_p,
                k=int(args.k),
                output_json=eval_p,
                output_jsonl=None if bool(args.no_eval_details) else eval_j,
                judge_backend=args.judge_backend,
            )

            metrics: Dict[str, Any] = {}
            if eval_p.exists():
                try:
                    metrics = json.loads(eval_p.read_text(encoding="utf-8"))
                except Exception:
                    metrics = {}

            summary["datasets"][ds.name] = {
                "data_path": str(ds.path),
                "tmp_parquet": str(tmp_p),
                "parquet": str(out_parquet),
                "eval_json": str(eval_p),
                "eval_details_jsonl": None if bool(args.no_eval_details) else str(eval_j),
                "metrics": metrics,
            }

            # Best-effort friendly one-liner
            passk = None
            try:
                passk = float(metrics.get("pass_at_k", {}).get("accuracy"))
            except Exception:
                passk = None
            if passk is not None:
                print(
                    f"[INFO] {ds.name}: pass@{int(args.k)}={passk:.4f} -> {eval_p}"
                )
            else:
                print(f"[INFO] {ds.name}: evaluated -> {eval_p}")

        _json_dump(summary, out_dir / "passk_summary.json")
        print(f"[INFO] Wrote summary: {out_dir / 'passk_summary.json'}")
        return

    # ---------------------
    # legacy backends
    # ---------------------

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gen_cfg = GenConfig(
        k=int(args.k),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=int(args.seed),
    )

    # Resolve ckpt list:
    #   --ckpts (CSV) > --ckpt (single) > env default list
    ckpt_list = _split_csv(args.ckpts)
    if not ckpt_list and args.ckpt:
        ckpt_list = [args.ckpt]
    if not ckpt_list:
        ckpt_list = _default_ckpts_from_env()
    if not ckpt_list:
        raise ValueError(
            "No checkpoint specified. Provide --ckpt/--ckpts or set $PRIMUS_SOURCE_CHECKPOINT_DIR."
        )

    multi_summary: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "backend": args.backend,
        "sampling": {
            "k": gen_cfg.k,
            "temperature": gen_cfg.temperature,
            "top_p": gen_cfg.top_p,
            "top_k": gen_cfg.top_k,
            "max_new_tokens": gen_cfg.max_new_tokens,
            "seed": gen_cfg.seed,
        },
        "models": {},
    }

    for ckpt in ckpt_list:
        ckpt = os.path.expandvars(ckpt)
        model_tag = _sanitize_tag(Path(ckpt.rstrip("/")).name or "model")
        out_dir = out_root / model_tag
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Model={model_tag} ckpt={ckpt}")

        # NOTE: vLLM doesn't need tokenizer for chat-template rendering, but we use it
        # to render list-of-messages prompts into a single string when possible.
        tokenizer = None
        if _HAS_TORCH:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    ckpt, trust_remote_code=args.trust_remote_code
                )
            except Exception:
                tokenizer = None

        if args.backend == "vllm":
            gen: Generator = VLLMGenerator(
                ckpt=ckpt,
                trust_remote_code=args.trust_remote_code,
                tensor_parallel_size=int(args.tensor_parallel_size),
                dtype=args.dtype,
                max_model_len=args.max_model_len,
            )
        else:
            gen = HFGenerator(
                ckpt=ckpt,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                device="cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu",
                max_prompt_tokens=args.max_prompt_tokens,
            )

        summary: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "ckpt": ckpt,
            "model_tag": model_tag,
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
                f"[INFO] {model_tag} / {ds.name}: pass@{gen_cfg.k}={metrics['pass@k']:.4f} "
                f"(n={metrics['num_examples']}) -> {metrics['output_jsonl']}"
            )

        _json_dump(summary, out_dir / "passk_summary.json")
        print(f"[INFO] Wrote summary: {out_dir / 'passk_summary.json'}")
        multi_summary["models"][model_tag] = {
            "ckpt": ckpt,
            "summary_path": str(out_dir / "passk_summary.json"),
        }

    if len(ckpt_list) > 1:
        _json_dump(multi_summary, out_root / "passk_summary.multi_models.json")
        print(f"[INFO] Wrote multi-model index: {out_root / 'passk_summary.multi_models.json'}")


if __name__ == "__main__":
    main()
