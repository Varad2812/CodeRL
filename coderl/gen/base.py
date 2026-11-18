"""Generator interface shared by the vLLM, Hugging Face and mock backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True)
class Completion:
    text: str
    num_tokens: int
    finish_reason: FinishReason  # "length" = hit max_new_tokens (truncated)


class Generator(ABC):
    """Sample completions for already chat-formatted prompts.

    Seeding contract: prompt ``i`` of a call is sampled with seed ``seed + i``, so callers that
    pass ``seed = base_seed + offset_of_first_prompt`` get per-problem seeds that do not depend
    on the batch size (exact for vLLM; the HF backend seeds once per internal batch).
    """

    name: str = "base"
    model_id: str = ""
    tokenizer: Any = None

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[Completion]]:
        """Return ``n`` completions per prompt, in prompt order."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any:
        return self.tokenizer.apply_chat_template(conversation, **kwargs)

    def info(self) -> dict[str, Any]:
        """Metadata recorded in results (backend, model, dtype, ...)."""
        return {"backend": self.name, "model": self.model_id}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Free accelerator memory."""
