from __future__ import annotations

from typing import Any

import pytest

from coderl.data.schema import Problem
from coderl.eval.prompts import SYSTEM_PROMPT, build_messages, render_prompt


def _p(source: str, prompt: str) -> Problem:
    return Problem(
        task_id=f"{source}/1",
        source=source,  # type: ignore[arg-type]
        split="test",
        prompt=prompt,
        entry_point="f",
        tests=["assert f() == 1"],
    )


class Tok:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return "|".join(f"{m['role']}:{m['content']}" for m in conversation) + "|assistant:"


def test_humaneval_message_has_signature_in_fence() -> None:
    msgs = build_messages(_p("humaneval", "def f():\n    '''Return one.'''\n"))
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert "```python\ndef f():\n    '''Return one.'''\n```" in msgs[1]["content"]
    assert "complete function" in msgs[1]["content"]


def test_mbpp_message_is_description_plus_example_assert() -> None:
    prompt = "Write f.\nYour code should pass this test:\nassert f() == 1"
    assert build_messages(_p("mbpp", prompt))[1]["content"] == prompt


def test_render_uses_chat_template_with_generation_prompt() -> None:
    tok = Tok()
    text = render_prompt(_p("mbpp", "Write f."), tok)
    assert text.endswith("|assistant:")
    assert tok.kwargs == {"tokenize": False, "add_generation_prompt": True}


def test_render_rejects_token_ids() -> None:
    class Bad(Tok):
        def apply_chat_template(self, conversation: list[dict[str, str]], **kw: Any) -> Any:
            return [1, 2, 3]

    with pytest.raises(TypeError):
        render_prompt(_p("mbpp", "x"), Bad())
