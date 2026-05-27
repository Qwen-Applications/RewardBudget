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
"""Multi-dataset generation entrypoint.

Why this exists:
  - The original `verl.trainer.main_generation` loads the checkpoint every run.
  - In RewardBudget evaluation, we often want to run multiple test datasets with the
    *same* checkpoint, and repeated checkpoint loading is very expensive.

This module loads the model once (initializes the Ray worker group once), then
iterates through multiple datasets sequentially.
"""

from __future__ import annotations

import os
from pprint import pprint

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "WARN")
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "true")


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation_multi(config)


def run_generation_multi(config) -> None:
    """Entry for CLI.

    Expected extra config fields:
      - data.paths: List[str]
      - data.output_paths: List[str]
    """
    if not ray.is_initialized():
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task_multi.remote(config))


@ray.remote(num_cpus=1)
def main_task_multi(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # -------- validate multi-dataset args --------
    paths = list(config.data.get("paths", []) or [])
    output_paths = list(config.data.get("output_paths", []) or [])
    if not paths or not output_paths:
        raise ValueError(
            "main_generation_multi requires config.data.paths and config.data.output_paths (non-empty lists)."
        )
    if len(paths) != len(output_paths):
        raise ValueError(
            f"Length mismatch: len(data.paths)={len(paths)} vs len(data.output_paths)={len(output_paths)}"
        )

    # -------- load tokenizer + init worker group ONCE --------
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    print("[INFO] Initializing model ONCE for multi-dataset generation...")
    wg.init_model()

    # -------- helper: generate for one parquet --------
    def _generate_one(data_path: str, output_path: str) -> None:
        dataset = pd.read_parquet(data_path)
        chat_lst = dataset[config.data.prompt_key].tolist()
        # parquet may store list-like
        chat_lst = [chat.tolist() for chat in chat_lst]

        total_samples = len(dataset)
        config_batch_size = config.data.batch_size
        apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        num_batch = -(-total_samples // config_batch_size)
        output_lst = [[] for _ in range(config.data.n_samples)]

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] ({os.path.basename(data_path)}) Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                **apply_chat_template_kwargs,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data = DataProto.from_dict(batch_dict)
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            print(f"[{batch_idx + 1}/{num_batch}] ({os.path.basename(data_path)}) Start to generate.")
            for n_sample in range(config.data.n_samples):
                output_padded = wg.generate_sequences(data_padded)
                output = unpad_dataproto(output_padded, pad_size=pad_size)

                output_texts = []
                for i in range(len(output)):
                    data_item = output[i]
                    prompt_length = data_item.batch["prompts"].shape[-1]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = data_item.batch["responses"][:valid_response_length]
                    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    output_texts.append(response_str)

                output_lst[n_sample].extend(output_texts)

        # (n_samples, n_data) -> (n_data, n_samples)
        output_arr = np.array(output_lst, dtype=object)
        output_arr = np.transpose(output_arr, axes=(1, 0)).tolist()
        dataset["responses"] = output_arr

        out_dir = os.path.dirname(output_path)
        makedirs(out_dir, exist_ok=True)
        dataset.to_parquet(output_path)

    # -------- run all datasets sequentially --------
    for i, (p, out_p) in enumerate(zip(paths, output_paths), start=1):
        print(f"[INFO] ({i}/{len(paths)}) Generating: {p} -> {out_p}")
        _generate_one(p, out_p)


if __name__ == "__main__":
    main()
