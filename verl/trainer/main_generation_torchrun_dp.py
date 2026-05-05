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

This file intentionally does not use Ray or model parallelism.  Launch it with
torchrun; each local rank loads a complete generation engine copy on its own GPU,
processes a round-robin shard of prompts, writes a rank-local shard file, and
rank 0 merges shard files by polling the shared file system.  By default it keeps
the historical HuggingFace Transformers backend.  Set ``+rollout.backend=vllm``
to use a rank-local vLLM offline engine for higher-throughput generation.

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
import socket
import time
from pathlib import Path
from pprint import pprint
from typing import Any
import inspect

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
    """Run shard-local generation under torchrun on a single node."""
    rank, world_size, local_rank = _torchrun_ranks()
    backend = _generation_backend(config)
    if backend == "vllm":
        # vLLM's offline engine otherwise tends to pick cuda:0 in every torchrun
        # process.  Restricting visible devices before CUDA is initialized makes
        # each torchrun rank behave like an independent single-GPU vLLM process.
        _prepare_rank_local_vllm_environment(config, local_rank=local_rank)
    else:
        _init_gloo_process_group(rank=rank, world_size=world_size)

    try:
        if rank == 0:
            pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        device = _select_device(0 if backend == "vllm" else local_rank)
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

        model = None
        llm = None
        if backend == "vllm":
            llm = _init_vllm_engine(config, model_path=model_path, trust_remote_code=trust_remote_code)
        else:
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
            if backend == "vllm":
                local_results = _generate_local_results_vllm(
                    local_positions=local_positions,
                    local_chats=local_chats,
                    tokenizer=tokenizer,
                    llm=llm,
                    config=config,
                    existing_responses_by_position=existing_responses,
                    shard_dir=shard_dir,
                    rank=rank,
                )
            else:
                local_results = _generate_local_results_hf(
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


def _to_plain_python(value: Any) -> Any:
    """Convert numpy/pyarrow scalar-or-array values from parquet to Python objects."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def _scalar_chat_field(value: Any) -> str:
    value = _to_plain_python(value)
    if isinstance(value, list):
        if len(value) == 0:
            return ""
        if len(value) == 1:
            return str(value[0])
        return "\n".join(str(item) for item in value)
    return "" if value is None else str(value)


def _normalize_chat(chat: Any) -> list[dict[str, str]]:
    """Normalize parquet prompt values to HF chat-template messages.

    The expected logical shape is list[dict(role, content)].  Depending on
    pyarrow/pandas versions, a parquet list<struct<role, content>> cell can be
    materialized as:
      - np.ndarray([{'role': 'user', 'content': '...'}])
      - [{'role': 'user', 'content': '...'}]
      - {'role': ['user'], 'content': ['...']}
      - {'role': 'user', 'content': '...'}

    Returning a bare dict is dangerous: tokenizer.apply_chat_template sees a
    batch like [dict, dict, ...] as one multi-turn conversation, so generation
    returns one output for many prompts.
    """
    chat = _to_plain_python(chat)

    if isinstance(chat, dict):
        role = _to_plain_python(chat.get("role", "user"))
        content = _to_plain_python(chat.get("content", ""))
        if isinstance(role, list) or isinstance(content, list):
            roles = role if isinstance(role, list) else [role] * len(content)
            contents = content if isinstance(content, list) else [content] * len(roles)
            if len(roles) != len(contents):
                raise ValueError(f"Malformed chat dict: len(role)={len(roles)} != len(content)={len(contents)}")
            return [
                {"role": _scalar_chat_field(r) or "user", "content": _scalar_chat_field(c)}
                for r, c in zip(roles, contents, strict=True)
            ]
        return [{"role": _scalar_chat_field(role) or "user", "content": _scalar_chat_field(content)}]

    if isinstance(chat, list):
        if len(chat) == 0:
            return [{"role": "user", "content": ""}]
        messages: list[dict[str, str]] = []
        for message in chat:
            message = _to_plain_python(message)
            if isinstance(message, dict):
                messages.append(
                    {
                        "role": _scalar_chat_field(message.get("role", "user")) or "user",
                        "content": _scalar_chat_field(message.get("content", "")),
                    }
                )
            else:
                messages.append({"role": "user", "content": _scalar_chat_field(message)})
        return messages

    return [{"role": "user", "content": _scalar_chat_field(chat)}]


def _chat_to_token_ids(
    *,
    tokenizer: Any,
    chat: list[dict[str, str]],
    config: DictConfig,
    apply_chat_template_kwargs: Any,
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        truncation=True,
        max_length=int(config.rollout.prompt_length),
        tokenize=True,
        **apply_chat_template_kwargs,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    ids = _to_plain_python(ids)
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise RuntimeError(f"Single chat unexpectedly encoded as a batch of {len(ids)} prompts.")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _batch_encode_chats_with_template(
    *,
    tokenizer: Any,
    chats: list[list[dict[str, str]]],
    config: DictConfig,
    apply_chat_template_kwargs: Any,
) -> Any:
    """Encode each chat independently, then pad.

    This avoids tokenizer-version-dependent ambiguity in batched
    apply_chat_template and guarantees input_ids.shape[0] == len(chats).
    """
    input_ids = [
        _chat_to_token_ids(
            tokenizer=tokenizer,
            chat=chat,
            config=config,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        for chat in chats
    ]
    return tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")


def _generation_backend(config: DictConfig) -> str:
    # Keep backward compatibility: main_generation_torchrun_dp historically
    # ignored rollout.name (which is often "vllm" in the shared generation
    # config) and used HF.  Only switch backend when explicitly requested.
    backend = config.rollout.get("backend", None)
    if backend is None:
        return "hf"
    backend_name = str(backend).lower()
    aliases = {"transformers": "hf", "huggingface": "hf"}
    backend_name = aliases.get(backend_name, backend_name)
    if backend_name not in {"hf", "vllm"}:
        raise ValueError(f"Unsupported rollout.backend={backend!r}; expected 'hf' or 'vllm'.")
    return backend_name


def _prepare_rank_local_vllm_environment(config: DictConfig, *, local_rank: int) -> None:
    """Make each torchrun rank look like an independent 1-GPU vLLM job.

    vLLM still initializes a torch.distributed process group internally, even
    for a single visible GPU.  If the torchrun/cluster MASTER_ADDR and
    MASTER_PORT leak into the vLLM EngineCore subprocess, vLLM can try to
    rendezvous with an unrelated host and hang in TCPStore.  Therefore the
    rank-local vLLM process gets a private loopback rendezvous endpoint.
    """
    if not bool(config.rollout.get("isolate_cuda_visible_devices", True)):
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if local_rank >= len(devices):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES={visible_devices!r}."
            )
        selected_device = devices[local_rank]
    else:
        selected_device = str(local_rank)

    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    # vLLM is run as an independent single-GPU offline engine inside each
    # torchrun process, so hide torchrun's global rank metadata from vLLM.
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_WORLD_SIZE"] = "1"

    # vLLM 0.11's UniProcExecutor ignores MASTER_PORT when choosing the
    # internal TCP rendezvous port; it uses VLLM_HOST_IP/VLLM_PORT instead.
    # Keep MASTER_* and VLLM_* in sync so the EngineCore does not inherit a
    # torchrun rendezvous endpoint or race on an implicit random port.
    rank_local_master_addr = str(config.rollout.get("rank_local_master_addr", "127.0.0.1"))
    rank_local_master_port = int(config.rollout.get("rank_local_master_port", _find_free_loopback_port()))
    os.environ["MASTER_ADDR"] = rank_local_master_addr
    os.environ["MASTER_PORT"] = str(rank_local_master_port)
    os.environ["VLLM_HOST_IP"] = rank_local_master_addr
    os.environ["VLLM_PORT"] = str(rank_local_master_port)

    # Clear vLLM data-parallel metadata in case it was set by the outer job.
    # This entrypoint implements data parallelism itself with torchrun shards;
    # every rank-local vLLM instance must remain a world-size-1 engine.
    os.environ["VLLM_DP_SIZE"] = "1"
    os.environ["VLLM_DP_RANK"] = "0"
    os.environ["VLLM_DP_RANK_LOCAL"] = "0"
    os.environ["VLLM_DP_MASTER_IP"] = rank_local_master_addr
    os.environ["VLLM_DP_MASTER_PORT"] = "0"

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if not bool(config.rollout.get("enable_v1_multiprocessing", False)):
        # Avoid spawning an extra EngineCore process inside every torchrun rank.
        # In vLLM 0.11 this nested multiprocessing path is the source of the
        # observed TCPStore client timeout on 127.0.0.1:<random_port>.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _vllm_dtype(dtype: Any) -> str:
    if dtype is None:
        return "auto"
    dtype_name = str(dtype).lower()
    mapping = {
        "auto": "auto",
        "none": "auto",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported rollout.dtype for vLLM torchrun generation: {dtype!r}")
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


def _generate_local_results_hf(
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
                inputs = _batch_encode_chats_with_template(
                    tokenizer=tokenizer,
                    chats=active_chats,
                    config=config,
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                )
                if inputs["input_ids"].shape[0] != len(active_chats):
                    raise RuntimeError(
                        f"Tokenizer encoded {inputs['input_ids'].shape[0]} prompts for "
                        f"{len(active_chats)} active chats. First chat={active_chats[0]!r}"
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


def _init_vllm_engine(config: DictConfig, *, model_path: str, trust_remote_code: bool) -> Any:
    tensor_parallel_size = int(config.rollout.get("tensor_model_parallel_size", 1))
    if tensor_parallel_size != 1:
        raise ValueError(
            "main_generation_torchrun_dp uses torchrun data parallelism with one rank-local "
            "vLLM engine per GPU. Set rollout.tensor_model_parallel_size=1, or use verl's "
            "Ray/vLLM rollout path for tensor-parallel generation."
        )

    try:
        from vllm import LLM
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as exc:
        raise ImportError(
            "rollout.backend=vllm requires vLLM to be installed in the runtime environment."
        ) from exc

    max_model_len = config.rollout.get("max_model_len", None)
    if max_model_len is None:
        max_model_len = int(config.rollout.prompt_length) + int(config.rollout.response_length)

    engine_kwargs: dict[str, Any] = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": trust_remote_code,
        "dtype": _vllm_dtype(config.rollout.get("dtype", "auto")),
        "tensor_parallel_size": tensor_parallel_size,
        # Force a rank-local single-worker executor.  Do not let vLLM reuse
        # torchrun's global rendezvous endpoint.
        "distributed_executor_backend": str(config.rollout.get("distributed_executor_backend", "uni")),
        "master_addr": str(config.rollout.get("master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1"))),
        "master_port": int(config.rollout.get("master_port", os.environ.get("MASTER_PORT", "29501"))),
        "gpu_memory_utilization": float(config.rollout.get("gpu_memory_utilization", 0.8)),
        "max_model_len": int(max_model_len),
        "max_num_batched_tokens": int(config.rollout.get("max_num_batched_tokens", 8192)),
        "max_num_seqs": int(config.rollout.get("max_num_seqs", 1024)),
        "enforce_eager": bool(config.rollout.get("enforce_eager", True)),
        "load_format": str(config.rollout.get("load_format", "auto")),
        "disable_log_stats": bool(config.rollout.get("disable_log_stats", True)),
        "enable_chunked_prefill": bool(config.rollout.get("enable_chunked_prefill", True)),
    }

    optional_engine_keys = (
        "enable_prefix_caching",
        "seed",
    )
    for key in optional_engine_keys:
        value = config.rollout.get(key, None)
        if value is not None:
            engine_kwargs[key] = value

    engine_kwargs_config = config.rollout.get("engine_kwargs", None)
    if engine_kwargs_config:
        engine_kwargs.update(_dict_config_to_plain_dict(engine_kwargs_config.get("vllm", {}) or {}))
    engine_kwargs = _filter_vllm_engine_kwargs(LLM, EngineArgs, engine_kwargs)
    return LLM(**engine_kwargs)


def _filter_vllm_engine_kwargs(llm_cls: Any, engine_args_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by this installed vLLM version.

    vLLM.LLM accepts **kwargs and forwards them to EngineArgs.  Filtering only
    against LLM.__init__ therefore lets unsupported EngineArgs fields, such as
    master_addr/master_port in vLLM 0.10/0.11, slip through and crash at
    EngineArgs(..., **kwargs).
    """
    allowed: set[str] = set()

    def add_signature_kwargs(callable_obj: Any) -> None:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }:
                allowed.add(name)

    # Explicit LLM constructor kwargs, e.g. model/tokenizer/dtype/trust_remote_code.
    add_signature_kwargs(llm_cls)

    # Extra **kwargs accepted by LLM must also be valid EngineArgs fields.
    try:
        import dataclasses

        if dataclasses.is_dataclass(engine_args_cls):
            allowed.update(field.name for field in dataclasses.fields(engine_args_cls) if field.init)
        else:
            add_signature_kwargs(engine_args_cls)
    except (TypeError, ValueError):
        add_signature_kwargs(engine_args_cls)

    if not allowed:
        return _filter_kwargs_for_callable(llm_cls, kwargs)

    dropped = sorted(key for key in kwargs if key not in allowed)
    if dropped:
        print(
            f"[vLLM] Dropping unsupported engine kwargs for this vLLM version: {dropped}",
            flush=True,
        )
    return {key: value for key, value in kwargs.items() if key in allowed}


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _dict_config_to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


def _vllm_sampling_params(config: DictConfig, *, n: int) -> Any:
    from vllm import SamplingParams

    temperature = float(config.rollout.temperature)
    sampling_kwargs: dict[str, Any] = {
        "n": int(n),
        "max_tokens": int(config.rollout.response_length),
        "temperature": temperature,
        "ignore_eos": bool(config.rollout.get("ignore_eos", False)),
        "detokenize": True,
        "skip_special_tokens": bool(config.rollout.get("skip_special_tokens", True)),
    }

    top_p = config.rollout.get("top_p", None)
    if top_p is not None:
        sampling_kwargs["top_p"] = float(top_p)

    top_k = config.rollout.get("top_k", None)
    if top_k is not None:
        top_k_int = int(top_k)
        sampling_kwargs["top_k"] = top_k_int if top_k_int > 0 else -1

    optional_sampling_keys = (
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "include_stop_str_in_output",
        "spaces_between_special_tokens",
        "seed",
    )
    for key in optional_sampling_keys:
        value = config.rollout.get(key, None)
        if value is not None:
            sampling_kwargs[key] = value

    sampling_kwargs = _filter_kwargs_for_callable(SamplingParams, sampling_kwargs)
    return SamplingParams(**sampling_kwargs)


def _apply_chat_template_to_prompt_ids(
    *,
    tokenizer: Any,
    chats: list[Any],
    config: DictConfig,
) -> list[list[int]]:
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    return [
        _chat_to_token_ids(
            tokenizer=tokenizer,
            chat=_normalize_chat(chat),
            config=config,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
        )
        for chat in chats
    ]


def _completion_text(tokenizer: Any, completion: Any) -> str:
    text = getattr(completion, "text", None)
    if text is not None:
        return text
    token_ids = getattr(completion, "token_ids", [])
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _generate_local_results_vllm(
    *,
    local_positions: list[int],
    local_chats: list[Any],
    tokenizer: Any,
    llm: Any,
    config: DictConfig,
    existing_responses_by_position: dict[int, list[str]] | None,
    shard_dir: Path,
    rank: int,
) -> list[dict[str, Any]]:
    if not local_positions:
        return []

    batch_size = int(config.data.batch_size)
    n_samples = int(config.data.n_samples)
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

    sampling_params_cache: dict[int, Any] = {}
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(pending_items))
        batch_items = pending_items[start:end]
        print(f"[local vLLM batch {batch_idx + 1}/{num_batches}] processing {len(batch_items)} prompts", flush=True)

        # Resumed shards may leave different numbers of missing samples per row.
        # Group by missing count so vLLM can still use n>1 for the common fresh-run case.
        missing_counts = sorted(
            {
                n_samples - len(responses_by_position.get(row_position, []))
                for row_position, _ in batch_items
            },
            reverse=True,
        )
        for missing_count in missing_counts:
            group_items = [
                (row_position, chat)
                for row_position, chat in batch_items
                if n_samples - len(responses_by_position.get(row_position, [])) == missing_count
            ]
            if not group_items:
                continue

            group_positions = [row_position for row_position, _ in group_items]
            group_chats = [chat for _, chat in group_items]
            prompt_token_ids = _apply_chat_template_to_prompt_ids(
                tokenizer=tokenizer,
                chats=group_chats,
                config=config,
            )
            vllm_inputs = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
            if missing_count not in sampling_params_cache:
                sampling_params_cache[missing_count] = _vllm_sampling_params(config, n=missing_count)

            outputs = llm.generate(
                prompts=vllm_inputs,
                sampling_params=sampling_params_cache[missing_count],
                use_tqdm=False,
            )
            for row_position, output in zip(group_positions, outputs, strict=True):
                completions = list(output.outputs)[:missing_count]
                if len(completions) < missing_count:
                    raise RuntimeError(
                        f"vLLM returned {len(completions)} completions for row_position={row_position}; "
                        f"expected {missing_count}."
                    )
                row_responses = responses_by_position.setdefault(row_position, [])
                for completion in completions:
                    row_responses.append(_completion_text(tokenizer, completion))
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
