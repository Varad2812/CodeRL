"""Hugging Face transformers backend: CPU, Apple MPS or CUDA. Slow but runs anywhere; used for
Mac smoke tests and as the fallback when vLLM is unavailable. Needs the ``hf`` extra."""

from __future__ import annotations

from typing import Any

from coderl.gen.base import Completion, FinishReason, Generator
from coderl.utils.logging import get_logger

log = get_logger(__name__)


def _pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(dtype: str, device: str) -> Any:
    import torch

    if dtype != "auto":
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            dtype
        ]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # MPS/CPU: fp32 avoids half-precision NaNs; models here are small


class HFGenerator(Generator):
    name = "hf"

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        device: str | None = None,
        adapter_path: str | None = None,
        max_batch_size: int = 32,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model
        self.adapter_path = adapter_path
        self.device = _pick_device(device)
        self.dtype = _pick_dtype(dtype, self.device)
        self.max_batch_size = max(1, max_batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, dtype=self.dtype
        ).to(self.device)  # type: ignore[arg-type]
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        eos = self.model.generation_config.eos_token_id
        ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        self.eos_ids = sorted(i for i in ids if i is not None)
        self._torch = torch
        log.info(
            "generator_ready",
            backend=self.name,
            model=model,
            device=self.device,
            dtype=str(self.dtype),
            adapter=adapter_path,
        )

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_id,
            "adapter": self.adapter_path,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "max_batch_size": self.max_batch_size,
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
        torch = self._torch
        per_chunk = max(1, self.max_batch_size // max(1, n))
        sampling = temperature > 0
        out: list[list[Completion]] = []
        for start in range(0, len(prompts), per_chunk):
            chunk = prompts[start : start + per_chunk]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(seed + start)
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": n,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.eos_ids,
                # override the model's generation_config (Qwen ships top_k=20, rep. penalty 1.05)
                "repetition_penalty": 1.0,
                "top_k": 0,
            }
            if sampling:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            with torch.inference_mode():
                seqs = self.model.generate(**enc, **kwargs)
            new = seqs[:, enc["input_ids"].shape[1] :].tolist()
            for i in range(len(chunk)):
                row = []
                for toks in new[i * n : (i + 1) * n]:
                    cut = next((j for j, t in enumerate(toks) if t in self.eos_ids), None)
                    finish: FinishReason = "length" if cut is None else "stop"
                    body = toks if cut is None else toks[:cut]
                    text = self.tokenizer.decode(body, skip_special_tokens=True)
                    ntok = len(toks) if cut is None else cut + 1
                    row.append(Completion(text=text, num_tokens=ntok, finish_reason=finish))
                out.append(row)
        return out

    def close(self) -> None:
        del self.model
        if self.device == "cuda":
            self._torch.cuda.empty_cache()
