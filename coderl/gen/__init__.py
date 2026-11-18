"""Text generation backends: ``get_generator("vllm" | "hf" | "auto", model, ...)``."""

from __future__ import annotations

from typing import Any

from coderl.gen.base import Completion, Generator
from coderl.utils.logging import get_logger

log = get_logger(__name__)

BACKENDS = ("vllm", "hf", "auto")


def vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:  # noqa: BLE001 - broken installs raise all sorts of errors
        return False
    return True


def get_generator(
    backend: str,
    model: str,
    *,
    dtype: str = "auto",
    adapter_path: str | None = None,
    fallback_to_hf: bool = True,
    vllm_kwargs: dict[str, Any] | None = None,
    hf_kwargs: dict[str, Any] | None = None,
) -> Generator:
    """Create a generator. ``auto`` = vLLM if importable, else HF. With ``fallback_to_hf``,
    a vLLM import or engine start-up failure falls back to HF with a WARNING."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown generator backend {backend!r}; expected one of {BACKENDS}")
    if backend in ("vllm", "auto"):
        if backend == "auto" and not vllm_importable():
            log.warning("generator_fallback", requested="auto", using="hf", reason="no vllm")
        else:
            try:
                from coderl.gen.vllm_backend import VLLMGenerator

                return VLLMGenerator(
                    model, dtype=dtype, adapter_path=adapter_path, **(vllm_kwargs or {})
                )
            except Exception as exc:
                if not fallback_to_hf:
                    raise
                log.warning(
                    "generator_fallback",
                    requested=backend,
                    using="hf",
                    reason=f"vLLM unavailable: {exc!r}"[:500],
                    hint="HF generation is much slower; see docs/KAGGLE.md",
                )
    from coderl.gen.hf import HFGenerator

    return HFGenerator(model, dtype=dtype, adapter_path=adapter_path, **(hf_kwargs or {}))


__all__ = ["BACKENDS", "Completion", "Generator", "get_generator", "vllm_importable"]
