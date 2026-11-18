"""Chat prompts for code generation (Qwen2.5-Coder-Instruct style chat template)."""

from __future__ import annotations

from typing import Any, Protocol

from coderl.data.schema import Problem

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write a correct, efficient solution. Return only one "
    "```python code block containing the complete function(s) and any needed imports, no "
    "explanation, no tests, no example usage."
)

Message = dict[str, str]


class ChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: list[Message], **kwargs: Any) -> Any: ...


def user_message(problem: Problem) -> str:
    """The user turn: HumanEval = signature + docstring; MBPP = description + example assert."""
    if problem.source == "humaneval":
        return (
            "Complete the following Python function. Return the complete function, including "
            "its signature and any imports it needs.\n\n"
            f"```python\n{problem.prompt.strip()}\n```"
        )
    return problem.prompt.strip()


def build_messages(problem: Problem) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(problem)},
    ]


def render_prompt(problem: Problem, tokenizer: ChatTokenizer) -> str:
    """Chat-formatted prompt string (generation prompt appended) for ``problem``."""
    text = tokenizer.apply_chat_template(
        build_messages(problem), tokenize=False, add_generation_prompt=True
    )
    if not isinstance(text, str):
        raise TypeError("apply_chat_template(tokenize=False) must return a string")
    return text
