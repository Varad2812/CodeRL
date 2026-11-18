"""Deterministic generator for tests: canned completions, no model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from coderl.data.schema import Problem
from coderl.eval.prompts import user_message
from coderl.gen.base import Completion, FinishReason, Generator

Responder = Callable[[str, int], str]  # (rendered prompt, sample index) -> completion text


class _MockTokenizer:
    def apply_chat_template(
        self, conversation: list[dict[str, str]], tokenize: bool = False, **_: Any
    ) -> str:
        body = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in conversation)
        return body + "<|assistant|>\n"


class MockGenerator(Generator):
    name = "mock"

    def __init__(self, responder: Responder, model_id: str = "mock") -> None:
        self.responder = responder
        self.model_id = model_id
        self.tokenizer = _MockTokenizer()
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int,
    ) -> list[list[Completion]]:
        self.calls.append(
            {"n_prompts": len(prompts), "n": n, "temperature": temperature, "seed": seed}
        )
        out = []
        for p in prompts:
            row = []
            for j in range(n):
                text = self.responder(p, j)
                ntok = len(text.split())
                finish: FinishReason = "length" if ntok >= max_new_tokens else "stop"
                row.append(Completion(text=text, num_tokens=ntok, finish_reason=finish))
            out.append(row)
        return out


def fenced(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def canonical_responder(
    problems: Sequence[Problem], wrong_every: int = 0, wrong: str = "def _nope():\n    pass\n"
) -> Responder:
    """Answer with each problem's canonical solution; if ``wrong_every`` > 0, samples whose
    index is a multiple of it are wrong instead (to exercise pass@k with 0 < c < n)."""
    by_message = {user_message(p): p for p in problems}

    def respond(prompt: str, j: int) -> str:
        prob = next((p for m, p in by_message.items() if m in prompt), None)
        if prob is None or (wrong_every and j % wrong_every == 0):
            return fenced(wrong)
        return "Here you go:\n" + fenced(prob.canonical_solution or "")

    return respond
