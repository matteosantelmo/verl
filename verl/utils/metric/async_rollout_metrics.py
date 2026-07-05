import time
import asyncio
import aiohttp
from dataclasses import asdict
from typing import Any, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import torch
import torch.distributed
from requests import get, post
from sglang.srt.utils import init_custom_process_group
from sglang.utils import wait_for_server
from torch.utils.data import DataLoader

from verl.utils.device import get_device_name
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.metric.utils import compute_token_ttr
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.metric.rollout_tasks import get_task, RolloutParams, compute_default_metrics
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

DEGENERATION_THRESHOLD = 0.9
MAX_NEW_TOKENS_WINDOW_SIZE = 1024
DEFAULT_SAMPLING_PARAMS = {
    "temperature": 0.0,
    "max_new_tokens": 256,
}
GLOBAL_DEFAULT_METRICS = {"token_ttr", "token_3gram_ttr", "text_ttr", "length"}


def _compute_metrics_worker(output: dict[str, Any], params: RolloutParams) -> dict[str, float]:
    metrics = {}
    task_name = params.task_name

    # Compute default metrics (TTR, length) for single outputs
    if isinstance(output, dict):
        default_metrics = compute_default_metrics(output)
        # Add global default metrics
        metrics.update(default_metrics)
        # Add per-benchmark default metrics
        if task_name:
            for key, value in default_metrics.items():
                metrics[f"{task_name}/{key}"] = value
    else:
        # For list of outputs, aggregate default metrics
        all_default_metrics = [compute_default_metrics(o) for o in output]
        # Aggregate global default metrics
        for key in all_default_metrics[0]:
            values = [m[key] for m in all_default_metrics]
            metrics[key] = sum(values) / len(values)
            # Aggregate per-benchmark default metrics
            if task_name:
                metrics[f"{task_name}/{key}"] = sum(values) / len(values)

    # Compute task-specific metrics
    task_ids = params.id.split(",")
    for task_id in task_ids:
        task_fn = get_task(task_id)
        if task_fn is not None:
            metrics.update(task_fn(output, params))

    return metrics


class AsyncRolloutMetrics:
    def __init__(
        self,
        rollout_dataset: MultiTurnSFTDataset,
        rollout_url: str,
        master_address: str = None,
        master_port: int = None,
        rollout_batch_size: int = 8192,
        pad_token_id: int = 3,
        num_workers: int = 128,
        standalone: bool = False,
    ):
        self.rollout_dataset = rollout_dataset
        self.rollout_url = rollout_url
        self.master_address = master_address
        self.master_port = master_port
        self.rollout_batch_size = rollout_batch_size
        self.pad_token_id = pad_token_id
        self.num_workers = num_workers
        self.standalone = standalone

        self._build_rollout_dataloader()

        if standalone:
            wait_for_server(rollout_url)
        else:
            self._init_weight_update_group()

        self.compute_metrics_step = 0
        self.compute_metrics_future = None

        self.process_pool = ProcessPoolExecutor(max_workers=num_workers)

        # Consume all the data to avoid having tokenizers issues
        self.batched_data = [batch for batch in self.rollout_dataloader]

    def _build_rollout_dataloader(self):
        self.rollout_dataloader = DataLoader(
            self.rollout_dataset,
            batch_size=self.rollout_batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True,
            pin_memory_device=get_device_name(),
        )

    def _init_weight_update_group(self):
        wait_for_server(self.rollout_url)

        response = get(
            url=f"{self.rollout_url}/workers",
        ).json()
        self.workers_urls = [worker["url"] for worker in response["workers"]]
        self.world_size = len(self.workers_urls) + 1

        thread_pool = ThreadPoolExecutor(len(self.workers_urls))

        futures = [thread_pool.submit(
            post,
            url=f"{worker_url}/init_weights_update_group",
            json={
                "master_address": self.master_address,
                "master_port": self.master_port,
                "rank_offset": i + 1,
                "world_size": self.world_size,
                "group_name": "weight_update_group",
                "backend": "nccl",
            },
        ) for i, worker_url in enumerate(self.workers_urls)]

        self.weight_update_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.master_port,
            world_size=self.world_size,
            is_master=True,
            timeout=torch.distributed.default_pg_timeout,
        )

        self.weight_update_group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_address}:{self.master_port}",
            world_size=self.world_size,
            rank=0,
            group_name="weight_update_group",
        )

        for future in futures:
            future.result()

    def update_rollout_engine(self, per_tensor_param: Iterator[tuple[str, torch.Tensor]]):
        start_time = time.time()

        bucket_bytes = 512 << 20
        for params_batch in get_named_tensor_buckets(per_tensor_param, bucket_bytes):
            names, dtypes, shapes = [], [], []
            for name, param in params_batch:
                names.append(name)
                dtypes.append(str(param.dtype).split(".")[-1])
                shapes.append(list(param.shape))

            thread_pool = ThreadPoolExecutor(len(self.workers_urls))

            futures = [thread_pool.submit(
                post,
                url=f"{worker_url}/update_weights_from_distributed",
                json={
                    "names": names,
                    "dtypes": dtypes,
                    "shapes": shapes,
                },
            ) for worker_url in self.workers_urls]

            handles = []
            for name, param in params_batch:
                handles.append(torch.distributed.broadcast(param, src=0, group=self.weight_update_group, async_op=True))

            for handle in handles:
                handle.wait()

            for future in futures:
                future.result()

        print(f"update_rollout_engine time: {time.time() - start_time}")

    def wait_compute_metrics(self) -> tuple[tuple[dict[str, float], list[dict[str, Any]]], int]:
        return (
            self.compute_metrics_future.result() if self.compute_metrics_future is not None else ({}, []),
            self.compute_metrics_step,
        )

    def async_compute_metrics(self, step: int, params: Iterator[tuple[str, torch.Tensor]]) -> tuple[tuple[dict[str, float], list[dict[str, Any]]], int]:
        start = time.time()
        output = self.wait_compute_metrics()
        print(f"wait_compute_metrics time: {time.time() - start}")

        self.update_rollout_engine(params)

        self.compute_metrics_step = step
        self.compute_metrics_future = ThreadPoolExecutor().submit(self._async_compute_metrics)
        return output

    def _get_sampling_params(self, params: RolloutParams) -> dict[str, Any]:
        sampling_params = DEFAULT_SAMPLING_PARAMS.copy()
        sampling_params.update(params.sampling_params)
        return sampling_params

    def _check_degeneration(self, output_ids: list[int]) -> bool:
        return (1 - compute_token_ttr(output_ids)) > DEGENERATION_THRESHOLD

    async def _async_generate_single(self, session: aiohttp.ClientSession, input_ids: list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        input_ids = input_ids.copy()
        sampling_params = sampling_params.copy()
        max_new_tokens = sampling_params["max_new_tokens"]

        aggregated_output = {"text": "", "output_ids": [], "meta_info": {"finish_reason": {"type": "null"}}}
        while max_new_tokens > 0:
            sampling_params["max_new_tokens"] = min(max_new_tokens, MAX_NEW_TOKENS_WINDOW_SIZE)
            async with session.post(
                url=f"{self.rollout_url}/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": sampling_params,
                },
            ) as response:
                if response.status != 200:
                    aggregated_output["meta_info"]["finish_reason"] = {"type": "error"}
                    return aggregated_output
                
                output = await response.json()
                aggregated_output["text"] += output["text"]
                aggregated_output["output_ids"] += output["output_ids"]
                aggregated_output["meta_info"] = output["meta_info"]

                if output["meta_info"]["finish_reason"]["type"] != "length":
                    return aggregated_output

                if self._check_degeneration(output["output_ids"]):
                    aggregated_output["meta_info"]["finish_reason"] = {"type": "degenerating"}
                    return aggregated_output
                
                max_new_tokens -= len(output["output_ids"])
                input_ids += output["output_ids"]

        return aggregated_output

    async def _async_generate(self, input_ids: list[int], sampling_params: dict[str, Any]) -> dict[str, Any]:
        n_samples = sampling_params.get("n", 1)
        base_params = sampling_params.copy()
        base_params.pop("n", None)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as session:
            tasks = [
                self._async_generate_single(session, input_ids, base_params)
                for _ in range(n_samples)
            ]
            return await asyncio.gather(*tasks)

    async def _async_generate_batch(
        self,
        unpadded_input_ids: list[list[int]],
        sampling_params_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tasks = [
            self._async_generate(input_ids, sampling_params)
            for input_ids, sampling_params in zip(unpadded_input_ids, sampling_params_list)
        ]

        return await asyncio.gather(*tasks)

    def _async_compute_metrics(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        metrics = {}
        clip_lengths = []
        clip_degenerations = []
        generations_data = []
        tokenizer = self.rollout_dataset.tokenizer
        for i, batch in enumerate(self.batched_data):
            clip_length = 0
            clip_degeneration = 0

            batch_params = [RolloutParams.from_dict(rollout_param) for rollout_param in batch["rollout_params"]]

            unpadded_input_ids = [ids[ids != self.pad_token_id].tolist() for ids in batch["input_ids"]]
            sampling_params = [self._get_sampling_params(params) for params in batch_params]

            input_texts = [tokenizer.decode(ids, skip_special_tokens=False) for ids in unpadded_input_ids]

            outputs = asyncio.run(self._async_generate_batch(unpadded_input_ids, sampling_params))

            len_outputs = 0
            for output in outputs:
                for o in (output if isinstance(output, list) else [output]):
                    if o["meta_info"]["finish_reason"]["type"] == "length":
                        clip_length += 1
                    elif o["meta_info"]["finish_reason"]["type"] == "degenerating":
                        clip_degeneration += 1
                    len_outputs += 1

            clip_length /= len_outputs
            clip_degeneration /= len_outputs

            # Submit metric computation tasks to separate processes
            futures = []
            for output, params in zip(outputs, batch_params):
                future = self.process_pool.submit(_compute_metrics_worker, output, params)
                futures.append((future, output, params))

            # Collect results and build detailed generation data
            for (future, output, params), sp, input_text in zip(futures, sampling_params, input_texts):
                new_metrics = future.result()
                for metric_name, metric_value in new_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)

                # Filter out global default metrics for per-generation logging
                filtered_metrics = {k: v for k, v in new_metrics.items() if k not in GLOBAL_DEFAULT_METRICS}

                # Build detailed generation data for each output sample
                output_list = output if isinstance(output, list) else [output]
                for o in output_list:

                    generation_data = {
                        "task_id": params.id,
                        "task_name": params.task_name,
                        "input": input_text,
                        "output": o["text"],
                        "finish_reason": o["meta_info"]["finish_reason"]["type"],
                        "metrics": filtered_metrics if filtered_metrics else {},
                        "sampling_params": sp,
                    }

                    for key, value in asdict(params).items():
                        if key not in ["id", "sampling_params", "task_name"] and isinstance(value, (str, int, float, bool)):
                            generation_data[key] = value

                    generations_data.append(generation_data)

            clip_lengths.append(clip_length)
            clip_degenerations.append(clip_degeneration)

        aggregated_metrics = {}
        for metric_name, values in metrics.items():
            if values:
                metric_name = f"rollout_{metric_name}" if metric_name not in GLOBAL_DEFAULT_METRICS else f"rollout/{metric_name}"
                aggregated_metrics[metric_name] = sum(values) / len(values)

        aggregated_metrics["rollout/clip_length"] = sum(clip_lengths) / len(clip_lengths)
        aggregated_metrics["rollout/clip_degeneration"] = sum(clip_degenerations) / len(clip_degenerations)

        return aggregated_metrics, generations_data

    def compute_metrics(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Compute metrics synchronously (for standalone mode)."""
        return self._async_compute_metrics()

    def shutdown(self):
        """Shutdown the metric computation process pool."""
        if hasattr(self, "process_pool"):
            self.process_pool.shutdown(wait=True)
        if hasattr(self, "bg_executor"):
            self.bg_executor.shutdown(wait=True)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.shutdown()
