# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Single-node torchrun data-parallel generation entrypoint.

This file intentionally does not use Ray, tensor parallelism, model parallelism, or
NCCL collectives.  Launch it with torchrun; each local rank loads a complete HF
model copy on its own GPU, processes a round-robin shard of prompts, writes a
rank-local shard file, and rank 0 merges shard files by polling the shared file
system.

Example:
    torchrun --standalone --nproc_per_node=8 -m verl.trainer.main_generation_torchrun_dp \
        data.path=/path/to/test.parquet \
        data.output_path=/path/to/output.parquet \
        model.path=/path/to/hf_model \
        data.batch_size=32 \
        data.n_samples=1
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from pprint import pprint
from typing import Any

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.import_utils import import_external_libs


os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "true")
os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config: DictConfig) -> None:
    run_generation_torchrun_dp(config)


def run_generation_torchrun_dp(config: DictConfig) -> None:
    """Run shard-local HF generation under torchrun on a single node."""
    rank, world_size, local_rank = _torchrun_ranks()
    _init_gloo_process_group(rank=rank, world_size=world_size)

    try:
        if rank == 0:
            pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        device = _select_device(local_rank)
        data_path = str(config.data.path)
        output_path = str(config.data.output_path)
        shard_dir = _build_shard_dir(output_path, world_size=world_size)
        ready_file = shard_dir / "_READY"

        if rank == 0:
            shard_dir.mkdir(parents=True, exist_ok=True)
            for done_file in shard_dir.glob("rank_*.done"):
                done_file.unlink()
            merged_file = shard_dir / "_MERGED"
            if merged_file.exists():
                merged_file.unlink()
            _atomic_write_text(ready_file, "ready\n")
        else:
            _wait_for_files([ready_file], timeout_sec=_merge_timeout_sec(config), poll_interval_sec=0.2)

        dataset = pd.read_parquet(data_path)
        total_rows = len(dataset)
        row_positions = list(range(total_rows))
        local_positions = shard_generation_requests(row_positions, rank=rank, world_size=world_size)
        local_dataset = dataset.iloc[local_positions]
        local_chats = [_normalize_chat(chat) for chat in local_dataset[config.data.prompt_key].tolist()]

        print(
            f"[rank {rank}/{world_size}] device={device} shard_size={len(local_positions)} "
            f"total_size={total_rows}",
            flush=True,
        )

        import_external_libs(config.model.get("external_lib", None))
        model_path = copy_to_local(str(config.model.path))
        trust_remote_code = bool(config.data.get("trust_remote_code", False))
        tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_torch_dtype(config.rollout.get("dtype", "auto")),
            trust_remote_code=trust_remote_code,
        )
        model.to(device)
        model.eval()

        if float(config.rollout.temperature) == 0.0 and int(config.data.n_samples) != 1:
            raise ValueError("When rollout.temperature=0, data.n_samples must be 1.")
        if int(config.data.n_samples) < 1:
            raise ValueError("data.n_samples must be >= 1.")

        n_samples = int(config.data.n_samples)
        existing_responses = _load_existing_rank_results(shard_dir, rank=rank, n_samples=n_samples)
        local_position_set = set(local_positions)
        existing_responses = {
            pos: responses for pos, responses in existing_responses.items() if pos in local_position_set
        }
        complete_existing = sum(1 for pos in local_positions if len(existing_responses.get(pos, [])) >= n_samples)
        if existing_responses:
            print(
                f"[rank {rank}/{world_size}] resumed {complete_existing}/{len(local_positions)} prompts "
                f"from {shard_dir / _rank_shard_name(rank)}",
                flush=True,
            )

        if complete_existing == len(local_positions):
            local_results = [
                {"row_position": row_position, "responses": existing_responses[row_position][:n_samples]}
                for row_position in local_positions
            ]
            _write_rank_shard(shard_dir, rank=rank, rows=local_results)
            print(f"[rank {rank}/{world_size}] reused shard with {len(local_results)} rows", flush=True)
        else:
            local_results = _generate_local_results(
                local_positions=local_positions,
                local_chats=local_chats,
                tokenizer=tokenizer,
                model=model,
                device=device,
                config=config,
                existing_responses_by_position=existing_responses,
                shard_dir=shard_dir,
                rank=rank,
            )
            _write_rank_shard(shard_dir, rank=rank, rows=local_results)
            print(f"[rank {rank}/{world_size}] wrote shard with {len(local_results)} rows", flush=True)

        if rank == 0:
            _merge_rank_shards(
                dataset=dataset,
                output_path=output_path,
                shard_dir=shard_dir,
                world_size=world_size,
                total_rows=total_rows,
                timeout_sec=_merge_timeout_sec(config),
                poll_interval_sec=_poll_interval_sec(config),
            )
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def shard_generation_requests(requests: list[Any], *, rank: int, world_size: int) -> list[Any]:
    """Return the rank-local round-robin prompt shard.

    rank 0 handles rows 0, world_size, 2 * world_size, ...; rank 1 handles rows
    1, 1 + world_size, ...; and so on. This keeps uneven prompt counts as
    balanced as possible while preserving a deterministic merge order.
    """
    if world_size <= 1:
        return list(requests)
    return list(requests[rank::world_size])


def _torchrun_ranks() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _init_gloo_process_group(*, rank: int, world_size: int) -> None:
    """Initialize torch.distributed without using NCCL or collectives."""
    if world_size <= 1 or not dist.is_available() or dist.is_initialized():
        return
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _select_device(local_rank: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("torchrun data-parallel generation expects CUDA to be available.")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def _normalize_chat(chat: Any) -> Any:
    """Convert parquet/list-like chat values to plain Python objects."""
    if hasattr(chat, "tolist"):
        return chat.tolist()
    return chat


def _torch_dtype(dtype: Any) -> Any:
    if dtype is None:
        return "auto"
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_name = str(dtype).lower()
    if dtype_name in {"auto", "none"}:
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported rollout.dtype for HF torchrun generation: {dtype!r}")
    return mapping[dtype_name]


def _build_shard_dir(output_path: str, *, world_size: int) -> Path:
    output = Path(output_path).expanduser()
    parent = output.parent if str(output.parent) else Path(".")
    basename = output.name or "generation_output.parquet"
    run_id = (
        os.environ.get("TORCHRUN_DP_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or f"{os.environ.get('MASTER_ADDR', 'localhost')}_{os.environ.get('MASTER_PORT', '0')}_{world_size}"
    )
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return parent / f".{basename}.torchrun_dp_shards.{safe_run_id}"


def _generation_kwargs(config: DictConfig, tokenizer: Any) -> dict[str, Any]:
    temperature = float(config.rollout.temperature)
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(config.rollout.response_length),
        "do_sample": bool(config.rollout.get("do_sample", temperature > 0.0)),
        "pad_token_id": tokenizer.pad_token_id,
    }

    if temperature > 0.0:
        kwargs["temperature"] = temperature
    else:
        kwargs["do_sample"] = False

    top_p = config.rollout.get("top_p", None)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)

    top_k = config.rollout.get("top_k", None)
    if top_k is not None and int(top_k) > 0:
        kwargs["top_k"] = int(top_k)

    if not bool(config.rollout.get("ignore_eos", False)) and tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id
    elif bool(config.rollout.get("ignore_eos", False)):
        kwargs["eos_token_id"] = None

    return kwargs


def _generate_local_results(
    *,
    local_positions: list[int],
    local_chats: list[Any],
    tokenizer: Any,
    model: AutoModelForCausalLM,
    device: torch.device,
    config: DictConfig,
    existing_responses_by_position: dict[int, list[str]] | None,
    shard_dir: Path,
    rank: int,
) -> list[dict[str, Any]]:
    if not local_positions:
        return []

    batch_size = int(config.data.batch_size)
    n_samples = int(config.data.n_samples)
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    generation_kwargs = _generation_kwargs(config, tokenizer)
    responses_by_position: dict[int, list[str]] = {
        row_position: list(responses[:n_samples])
        for row_position, responses in (existing_responses_by_position or {}).items()
    }
    pending_items = [
        (row_position, chat)
        for row_position, chat in zip(local_positions, local_chats, strict=True)
        if len(responses_by_position.get(row_position, [])) < n_samples
    ]
    num_batches = -(-len(pending_items) // batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(pending_items))
        batch_items = pending_items[start:end]
        batch_positions = [row_position for row_position, _ in batch_items]
        batch_chats = [chat for _, chat in batch_items]
        print(f"[local batch {batch_idx + 1}/{num_batches}] processing {len(batch_chats)} prompts", flush=True)

        with torch.inference_mode():
            for sample_idx in range(n_samples):
                active_indices = [
                    row_idx
                    for row_idx, row_position in enumerate(batch_positions)
                    if len(responses_by_position.get(row_position, [])) <= sample_idx
                ]
                if not active_indices:
                    continue

                active_chats = [batch_chats[row_idx] for row_idx in active_indices]
                inputs = tokenizer.apply_chat_template(
                    active_chats,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=int(config.rollout.prompt_length),
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                    **apply_chat_template_kwargs,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                input_length = inputs["input_ids"].shape[-1]
                output_ids = model.generate(**inputs, **generation_kwargs)
                response_ids = output_ids[:, input_length:]
                responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                for row_idx, response in zip(active_indices, responses, strict=True):
                    row_position = batch_positions[row_idx]
                    row_responses = responses_by_position.setdefault(row_position, [])
                    row_responses.append(response)
                    _append_rank_shard_row(
                        shard_dir,
                        rank=rank,
                        row={"row_position": row_position, "responses": row_responses},
                    )

    incomplete = [
        row_position
        for row_position in local_positions
        if len(responses_by_position.get(row_position, [])) < n_samples
    ]
    if incomplete:
        preview = incomplete[:20]
        raise RuntimeError(f"Incomplete generated rows: {preview} (total_incomplete={len(incomplete)})")

    return [
        {"row_position": row_position, "responses": responses_by_position[row_position][:n_samples]}
        for row_position in local_positions
    ]


def _write_rank_shard(shard_dir: Path, *, rank: int, rows: list[dict[str, Any]]) -> None:
    shard_path = shard_dir / _rank_shard_name(rank)
    tmp_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, shard_path)
    _atomic_write_text(shard_dir / _rank_done_name(rank), "done\n")


def _append_rank_shard_row(shard_dir: Path, *, rank: int, row: dict[str, Any]) -> None:
    shard_path = shard_dir / _rank_shard_name(rank)
    with shard_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_existing_rank_results(shard_dir: Path, *, rank: int, n_samples: int) -> dict[int, list[str]]:
    shard_path = shard_dir / _rank_shard_name(rank)
    if not shard_path.exists():
        return {}

    responses_by_position: dict[int, list[str]] = {}
    with shard_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Ignoring invalid JSON line in {shard_path}:{line_no}", flush=True)
                continue

            try:
                row_position = int(row["row_position"])
                responses = row["responses"]
                if not isinstance(responses, list):
                    raise TypeError
            except (KeyError, TypeError, ValueError):
                print(f"Ignoring malformed generation row in {shard_path}:{line_no}", flush=True)
                continue

            responses = responses[:n_samples]
            if len(responses) >= len(responses_by_position.get(row_position, [])):
                responses_by_position[row_position] = responses

    return responses_by_position


def _merge_rank_shards(
    *,
    dataset: pd.DataFrame,
    output_path: str,
    shard_dir: Path,
    world_size: int,
    total_rows: int,
    timeout_sec: float,
    poll_interval_sec: float,
) -> None:
    shard_paths = [shard_dir / _rank_shard_name(rank) for rank in range(world_size)]
    done_paths = [shard_dir / _rank_done_name(rank) for rank in range(world_size)]
    _wait_for_files(done_paths, timeout_sec=timeout_sec, poll_interval_sec=poll_interval_sec)

    responses_by_position: dict[int, list[str]] = {}
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_position = int(row["row_position"])
                if row_position in responses_by_position:
                    raise ValueError(f"Duplicate generated row_position while merging shards: {row_position}")
                responses_by_position[row_position] = list(row["responses"])

    missing = [idx for idx in range(total_rows) if idx not in responses_by_position]
    if missing:
        preview = missing[:20]
        raise RuntimeError(f"Missing generated rows from rank shards: {preview} (total_missing={len(missing)})")

    dataset["responses"] = [responses_by_position[idx] for idx in range(total_rows)]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(output_path)
    _atomic_write_text(shard_dir / "_MERGED", f"{output_path}\n")
    print(f"[rank 0] merged {total_rows} rows to {output_path}", flush=True)


def _rank_shard_name(rank: int) -> str:
    return f"rank_{rank:05d}.jsonl"


def _rank_done_name(rank: int) -> str:
    return f"rank_{rank:05d}.done"


def _wait_for_files(paths: list[Path], *, timeout_sec: float, poll_interval_sec: float) -> None:
    start_time = time.monotonic()
    missing = [path for path in paths if not path.exists()]
    while missing:
        if time.monotonic() - start_time > timeout_sec:
            missing_str = ", ".join(str(path) for path in missing[:10])
            raise TimeoutError(f"Timed out waiting for shard files: {missing_str}")
        time.sleep(poll_interval_sec)
        missing = [path for path in paths if not path.exists()]


def _merge_timeout_sec(config: DictConfig) -> float:
    return float(config.data.get("merge_timeout_sec", 24 * 60 * 60))


def _poll_interval_sec(config: DictConfig) -> float:
    return float(config.data.get("shard_poll_interval_sec", 5.0))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
