"""vLLM backend (Linux + CUDA, the ``gen`` extra).

T4 notes: no bfloat16 -> pass ``dtype="float16"``; FlashAttention-2 is unavailable on SM75 and
vLLM picks a Triton attention backend on its own. ``tensor_parallel_size=2`` spreads one model
over Kaggle's two T4s; for two small models it is simpler to run one process per GPU
(``CUDA_VISIBLE_DEVICES``), which is what the Kaggle notebook does.
"""

from __future__ import annotations

from typing import Any

from coderl.gen.base import Completion, Generator
from coderl.utils.logging import get_logger

log = get_logger(__name__)


class VLLMGenerator(Generator):
    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 2048,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        adapter_path: str | None = None,
        enforce_eager: bool = False,
        seed: int = 0,
        revision: str | None = None,
    ) -> None:
        from vllm import LLM

        self.model_id = model
        self.adapter_path = adapter_path
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        lora = enable_lora or adapter_path is not None
        kwargs: dict[str, Any] = {
            "model": model,
            "revision": revision,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
            "seed": seed,
            "enable_lora": lora,
        }
        if lora:
            kwargs["max_lora_rank"] = max_lora_rank
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._lora_request: Any = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)
        log.info(
            "generator_ready",
            backend=self.name,
            model=model,
            dtype=dtype,
            tp=tensor_parallel_size,
            adapter=adapter_path,
        )

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_id,
            "adapter": self.adapter_path,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
        }

    def generate(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[Completion]]:
        from vllm import SamplingParams

        params = [
            SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p if temperature > 0 else 1.0,
                max_tokens=max_new_tokens,
                repetition_penalty=1.0,
                seed=seed + i,
            )
            for i in range(len(prompts))
        ]
        outputs = self.llm.generate(
            prompts, params, lora_request=self._lora_request, use_tqdm=False
        )
        return [
            [
                Completion(
                    text=o.text,
                    num_tokens=len(o.token_ids),
                    finish_reason="length" if o.finish_reason == "length" else "stop",
                )
                for o in out.outputs
            ]
            for out in outputs
        ]
