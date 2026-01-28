#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run inference on one or more datasets and report pass@k.

This script is designed for "bring-your-own-checkpoint" evaluation.

IMPORTANT (hard-coded model loading):
  This script ALWAYS loads exactly two models from the environment variable
  PRIMUS_SOURCE_CHECKPOINT_DIR:

    ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B
    ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Llama-3.2-3B-Instruct

  Any CLI args like --ckpt / --ckpts are accepted for backward compatibility
  but will be ignored (a warning is printed if they are provided).

Outputs:
  - Legacy backends (vllm/hf):
      <output_dir>/<model_tag>/<dataset>_powers.generations.jsonl
      <output_dir>/<model_tag>/passk_summary.json
      and (if 2+ models) <output_dir>/passk_summary.multi_models.json

  - verl backend (matching recipe/GARO/scripts/evaluate/eval_all.sh):
      <output_dir>/GARO/<dataset>_<run_tag>.parquet
      <output_dir>/GARO/<dataset>_<run_tag>.eval.json
      <output_dir>/GARO/<dataset>_<run_tag>.eval.details.jsonl (optional)
      <output_dir>/GARO/passk_summary.json (aggregated over the two models)

Dataset resolution heuristic:
  1) If a dataset name is already a file path, use it directly.
  2) Prefer GARO-style layout if --data_root is provided:
       <data_root>/GARO/test_data/<name>.parquet (or .jsonl/.json)
  3) Fall back to built-in:
       "math500" -> llm_experiments/data/MATH500.json

Note: verl backend requires parquet inputs (same as eval_all.sh).
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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

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


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _expand_path_str(x: str) -> str:
    """Expand env vars + ~ and return a string."""
    return os.path.expandvars(os.path.expanduser(x))


def _expand_path(x: Union[str, Path]) -> Path:
    """Expand env vars + ~ and return a Path (not necessarily exists)."""
    p = Path(_expand_path_str(str(x)))
    return p


def _sanitize_tag(s: str) -> str:
    s = s.strip()
    if not s:
        return "run"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s.strip("._-") or "run"


def _default_ckpts_from_env() -> List[str]:
    """
    Hard-coded: load exactly two models from ${PRIMUS_SOURCE_CHECKPOINT_DIR}:
      ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B
      ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Llama-3.2-3B-Instruct
    """
    base = os.environ.get("PRIMUS_SOURCE_CHECKPOINT_DIR")
    if not base:
        return []
    base = _expand_path_str(base)
    return [
        str(Path(base) / "Qwen3-8B"),
        str(Path(base) / "Llama-3.2-3B-Instruct"),
    ]


def _validate_ckpt_dir(ckpt: str) -> None:
    """
    verl/vllm/transformers expect a *real local HF model directory*.
    If ckpt is a base dir like /primus_source_checkpoint (not a model dir),
    transformers may treat it like a hub repo id and throw HFValidationError.
    """
    p = _expand_path(ckpt)
    # We validate by checking config.json (most reliable signal).
    cfg = p / "config.json"
    if not cfg.exists():
        raise FileNotFoundError(
            f"Invalid checkpoint path (missing config.json): {p}\n"
            "This must be a local HuggingFace model directory.\n"
            "Expected something like:\n"
            "  ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B\n"
            "  ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Llama-3.2-3B-Instruct\n"
        )


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
    rm = ex.get("reward_model")
    rm = _maybe_parse_json(rm)
    if isinstance(rm, dict):
        gt = rm.get("ground_truth") or rm.get("answer") or rm.get("gt")
        if gt is not None:
            return str(gt)

    for k in ("answer", "ground_truth", "gt", "solution"):
        if k in ex and ex[k] is not None:
            return str(ex[k])

    return None


def _prompt_to_text(prompt: Any, tokenizer: Optional[Any]) -> str:
    if isinstance(prompt, str):
        return prompt

    if isinstance(prompt, list):
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in prompt):
                try:
                    return tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass

        try:
            return "\n".join(
                m.get("content", str(m)) if isinstance(m, dict) else str(m) for m in prompt
            )
        except Exception:
            return str(prompt)

    return str(prompt)


def _extract_final_answer(text: str) -> str:
    boxed = parse_answer(text)
    if boxed is not None:
        return boxed.strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]

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
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    os.environ.setdefault("VLLM_USE_NCCL_SYMM_MEM", "0")
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    os.environ.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "1")


def _hydra_list_str(items: Sequence[Union[str, Path]]) -> str:
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
    if temperature == 0.0 and n_samples != 1:
        raise ValueError(
            "verl.trainer.main_generation_multi asserts: when temperature=0, n_samples must be 1. "
            "Set --temperature > 0 to sample multiple responses (k>1)."
        )

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


def _normalize_dataset_name(name: str) -> str:
    return name.strip()


def resolve_datasets(dataset_arg: str, data_root: Optional[str], repo_root: Path) -> List[DatasetSpec]:
    names = [_normalize_dataset_name(x) for x in dataset_arg.split(",") if x.strip()]
    specs: List[DatasetSpec] = []

    builtins = {
        "math500": repo_root / "llm_experiments" / "data" / "MATH500.json",
        "MATH500": repo_root / "llm_experiments" / "data" / "MATH500.json",
    }

    for name in names:
        p = _expand_path(name)
        if p.exists() and p.is_file():
            specs.append(DatasetSpec(name=p.stem, path=p))
            continue

        tried: List[Path] = []
        if data_root is not None:
            cand_dir = _expand_path(data_root) / "GARO" / "test_data"
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

        if name in builtins and builtins[name].exists():
            specs.append(DatasetSpec(name=name.lower(), path=builtins[name]))
            continue

        if data_root is None:
            raise FileNotFoundError(
                f"Cannot resolve dataset '{name}'. Provide --data_root, or pass a file path."
            )

        raise FileNotFoundError(
            f"Cannot find dataset file for '{name}'. Tried: {', '.join(map(str, tried))}"
        )

    return specs


def iter_examples(path: Path, batch_size: int = 256) -> Iterator[Dict[str, Any]]:
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

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=trust_remote_code)
        torch_dtype = None
        if dtype:
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

            prompt_len = toks["input_ids"].shape[1]
            gens: List[str] = []
            for seq in gen_ids:
                gen_text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                gens.append(gen_text)
            all_gens.append(gens)

        return all_gens


# -----------------------------
# Main evaluation loop (legacy)
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

    tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_f = tmp_jsonl.open("w", encoding="utf-8")

    prompt_buf: List[str] = []
    raw_buf: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal total, passk_hits, prompt_buf, raw_buf
        if not prompt_buf:
            return

        gens = gen.generate_many(prompt_buf, cfg)
        assert len(gens) == len(raw_buf)

        for ex, prompt_text, ex_gens in zip(raw_buf, prompt_buf, gens):
            gt = _extract_ground_truth(ex) or ""

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
        prompt = _maybe_parse_json(ex.get("prompt"))
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

    _json_dump(metrics, out_dir / f"{ds_tag}.metrics.json")
    return metrics


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on one or more datasets and compute pass@k."
    )

    # kept for backward compatibility (IGNORED)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "IGNORED. This script always loads exactly two models from "
            "$PRIMUS_SOURCE_CHECKPOINT_DIR: Qwen3-8B and Llama-3.2-3B-Instruct."
        ),
    )
    parser.add_argument(
        "--ckpts",
        type=str,
        default=None,
        help="IGNORED. Kept for backward-compatibility.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default="math500,amc23,aime24,aime25,omni_math,minerva,OlympiadBench_math_en_comp,OlympiadBench_physics_en_comp",
        help="Comma-separated dataset names (or file paths).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/primus_datasets/zmy",
        help=(
            "Root directory for GARO-style datasets (expects GARO/test_data/<name>.parquet). "
            "Defaults to /primus_datasets/zmy."
        ),
    )
    parser.add_argument("-k", type=int, default=1, help="Number of samples per prompt.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="${OUTPUT_DIR}",
        help=(
            "Output directory (parent). Can contain env vars (e.g. ${OUTPUT_DIR}). "
            "For verl backend, outputs are written under <output_dir>/GARO. "
            "For legacy backends, jsonl + summary are written under <output_dir>/<model_tag>/."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["verl", "vllm", "hf"],
        default=(
            "verl"
            if (Path(__file__).resolve().parents[2] / "verl").exists()
            else ("vllm" if _HAS_VLLM else "hf")
        ),
        help=(
            "Generation backend. 'verl' matches recipe/GARO/scripts/evaluate/eval_all.sh "
            "(calls verl.trainer.main_generation_multi and recipe evaluation)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for legacy generation (prompts per batch).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Max new tokens per generation (legacy backends).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy/deterministic).",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for deterministic sampling).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to HF/vLLM/verl.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size (vLLM) / rollout.tensor_model_parallel_size (verl).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="",
        help="dtype for vLLM/HF (e.g. float16, bfloat16). Empty = default.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="vLLM max_model_len (optional).",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=None,
        help="HF-only: truncate prompt to this many tokens (optional).",
    )
    parser.add_argument(
        "--write_prompt_text",
        action="store_true",
        help="Also store rendered prompt_text in jsonl (can be large).",
    )

    # ---- verl (eval_all.sh-compatible) knobs ----
    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help=(
            "Base tag used in output filenames for verl backend. The actual per-model "
            "tag becomes '<model_tag>_<base_tag>' to avoid collisions."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model name used to build default base_tag (verl backend only).",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="",
        help="Optional step used to build default base_tag (verl backend only).",
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
        help="Parquet column name containing prompts (verl backend).",
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="trainer.nnodes for verl generation (default: 1).",
    )
    parser.add_argument(
        "--n_gpus_per_node",
        type=int,
        default=1,
        help="trainer.n_gpus_per_node for verl generation (default: 1).",
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
        default="qwen_primus",
        choices=["qwen_primus", "openai_compat"],
        help="Judge backend passed to recipe/GARO/scripts/evaluate/evaluation.py (verl backend).",
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

    # Always evaluate exactly these two models (hard-coded by requirement).
    ckpt_list = _default_ckpts_from_env()
    if not ckpt_list:
        raise ValueError(
            "PRIMUS_SOURCE_CHECKPOINT_DIR is not set (or empty).\n"
            "Please set it so that the following paths exist:\n"
            "  ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Qwen3-8B\n"
            "  ${PRIMUS_SOURCE_CHECKPOINT_DIR}/Llama-3.2-3B-Instruct\n"
            f"Current PRIMUS_SOURCE_CHECKPOINT_DIR={os.environ.get('PRIMUS_SOURCE_CHECKPOINT_DIR')!r}"
        )

    # Strong validation: prevents accidentally passing base dir like /primus_source_checkpoint.
    for ck in ckpt_list:
        _validate_ckpt_dir(ck)

    if args.ckpt or args.ckpts:
        print(
            "[WARN] --ckpt/--ckpts are ignored. Using PRIMUS_SOURCE_CHECKPOINT_DIR hard-coded models."
        )

    # Expand output dir correctly (and do NOT bake in /GARO twice).
    out_root = _expand_path(args.output_dir)
    if "${" in str(out_root) or "$" in str(out_root):
        # This means user passed an env-var-like string but it didn't expand (env missing).
        # We'll fail early with a clear message.
        raise ValueError(
            f"--output_dir contains an env var that did not expand: {args.output_dir!r}\n"
            "Please export the environment variable(s) referenced by --output_dir."
        )
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------
    # verl backend
    # ---------------------
    if args.backend == "verl":
        if not (project_root / "verl").exists():
            raise RuntimeError(
                "Backend 'verl' selected but 'verl/' package not found under project root. "
                "Run from the repo root or use --backend vllm/hf."
            )

        non_parquet = [ds for ds in datasets if ds.path.suffix.lower() != ".parquet"]
        if non_parquet:
            bad = ", ".join(f"{ds.name}={ds.path}" for ds in non_parquet)
            raise ValueError(
                "verl backend only supports parquet inputs (same as eval_all.sh). "
                f"Got non-parquet dataset(s): {bad}"
            )

        out_dir = out_root / "GARO"
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = _expand_path(args.tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        base_tag = (args.run_tag or "").strip()
        if not base_tag:
            if (args.model or "").strip() and (args.step or "").strip():
                base_tag = f"{args.model.strip()}_{args.step.strip()}"
            elif (args.model or "").strip():
                base_tag = args.model.strip()
            else:
                base_tag = "run"
        base_tag = _sanitize_tag(base_tag)

        data_paths = [ds.path for ds in datasets]

        multi_summary: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "backend": "verl",
            "base_tag": base_tag,
            "models": {},
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
            "datasets": [ds.name for ds in datasets],
        }

        for ckpt in ckpt_list:
            ckpt = _expand_path_str(ckpt)
            _validate_ckpt_dir(ckpt)

            model_tag = _sanitize_tag(Path(ckpt.rstrip("/")).name or "model")
            run_tag = _sanitize_tag(f"{model_tag}_{base_tag}" if base_tag else model_tag)

            tmp_paths = [tmp_dir / f"{ds.name}_{run_tag}.parquet" for ds in datasets]
            eval_paths = [out_dir / f"{ds.name}_{run_tag}.eval.json" for ds in datasets]
            eval_jsonl_paths = [
                (out_dir / f"{ds.name}_{run_tag}.eval.details.jsonl") for ds in datasets
            ]

            print(
                f"[INFO] Backend=verl | model={model_tag} | run_tag={run_tag} | out_dir={out_dir} | tmp_dir={tmp_dir}"
            )
            print(f"[INFO] Checkpoint path: {ckpt}")
            print(f"[INFO] Data path(s): {' '.join(map(str, data_paths))}")

            _run_verl_generation_multi(
                project_root=project_root,
                ckpt=ckpt,
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

            model_summary: Dict[str, Any] = {
                "ckpt": ckpt,
                "model_tag": model_tag,
                "run_tag": run_tag,
                "datasets": {},
            }

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

                model_summary["datasets"][ds.name] = {
                    "data_path": str(ds.path),
                    "tmp_parquet": str(tmp_p),
                    "parquet": str(out_parquet),
                    "eval_json": str(eval_p),
                    "eval_details_jsonl": None if bool(args.no_eval_details) else str(eval_j),
                    "metrics": metrics,
                }

                passk = None
                try:
                    passk = float(metrics.get("pass_at_k", {}).get("accuracy"))
                except Exception:
                    passk = None
                if passk is not None:
                    print(f"[INFO] {model_tag} / {ds.name}: pass@{int(args.k)}={passk:.4f} -> {eval_p}")
                else:
                    print(f"[INFO] {model_tag} / {ds.name}: evaluated -> {eval_p}")

            multi_summary["models"][model_tag] = model_summary

        _json_dump(multi_summary, out_dir / "passk_summary.json")
        print(f"[INFO] Wrote summary: {out_dir / 'passk_summary.json'}")
        return

    # ---------------------
    # legacy backends
    # ---------------------

    gen_cfg = GenConfig(
        k=int(args.k),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        seed=int(args.seed),
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
        ckpt = _expand_path_str(ckpt)
        _validate_ckpt_dir(ckpt)

        model_tag = _sanitize_tag(Path(ckpt.rstrip("/")).name or "model")
        model_out_dir = out_root / model_tag
        model_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Backend={args.backend} | Model={model_tag} ckpt={ckpt}")

        tokenizer = None
        if _HAS_TORCH:
            try:
                tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=args.trust_remote_code)
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
                out_dir=model_out_dir,
                batch_size=int(args.batch_size),
                write_prompt_text=bool(args.write_prompt_text),
            )
            summary["datasets"][ds.name] = metrics
            print(
                f"[INFO] {model_tag} / {ds.name}: pass@{gen_cfg.k}={metrics['pass@k']:.4f} "
                f"(n={metrics['num_examples']}) -> {metrics['output_jsonl']}"
            )

        _json_dump(summary, model_out_dir / "passk_summary.json")
        print(f"[INFO] Wrote summary: {model_out_dir / 'passk_summary.json'}")
        multi_summary["models"][model_tag] = {
            "ckpt": ckpt,
            "summary_path": str(model_out_dir / "passk_summary.json"),
        }

    if len(ckpt_list) > 1:
        _json_dump(multi_summary, out_root / "passk_summary.multi_models.json")
        print(f"[INFO] Wrote multi-model index: {out_root / 'passk_summary.multi_models.json'}")


if __name__ == "__main__":
    main()
