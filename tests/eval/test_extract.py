"""Extraction on realistic (messy) model outputs."""

from __future__ import annotations

import pytest

from coderl.data.schema import Problem
from coderl.eval.extract import extract_code, strip_test_code

HE_PROMPT = (
    "from typing import List\n\n\n"
    "def has_close(numbers: List[float], threshold: float) -> bool:\n"
    '    """ Any two numbers closer than threshold?\n'
    "    >>> has_close([1.0, 2.0], 0.5)\n    False\n"
    '    """\n'
)
HE_HELPER_PROMPT = (
    "\n\ndef is_palindrome(string: str) -> bool:\n"
    '    """ Test if given string is a palindrome """\n'
    "    return string == string[::-1]\n\n\n"
    "def make_palindrome(string: str) -> str:\n"
    '    """ Shortest palindrome starting with string. """\n'
)


def he(prompt: str = HE_PROMPT, entry: str = "has_close") -> Problem:
    return Problem(
        task_id="humaneval/x",
        source="humaneval",
        split="test",
        prompt=prompt,
        entry_point=entry,
        tests=["assert True"],
    )


def mbpp(entry: str = "add") -> Problem:
    return Problem(
        task_id="mbpp/x",
        source="mbpp",
        split="test",
        prompt="Write a function to add.\nYour code should pass this test:\nassert add(1, 2) == 3",
        entry_point=entry,
        tests=["assert add(1, 2) == 3"],
    )


def run(code: str, call: str) -> object:
    ns: dict[str, object] = {}
    exec(code, ns)  # trusted test strings only
    return eval(call, ns)


def test_single_clean_block() -> None:
    out = "```python\ndef add(a, b):\n    return a + b\n```"
    ex = extract_code(out, mbpp())
    assert ex.status == "code_block" and ex.notes == []
    assert run(ex.code, "add(1, 2)") == 3


def test_prose_around_block() -> None:
    out = (
        "Sure! Here is the solution:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\n"
        "This function adds two numbers. Time complexity: O(1)."
    )
    ex = extract_code(out, mbpp())
    assert ex.status == "code_block"
    assert "Sure" not in ex.code and "complexity" not in ex.code


def test_multiple_blocks_prefers_the_one_defining_entry_point() -> None:
    out = (
        "First an example:\n```python\nprint(add(1, 2))\n```\n"
        "Implementation:\n```python\ndef add(a, b):\n    return a + b\n```\n"
        "Tests:\n```python\nassert add(1, 2) == 3\n```"
    )
    ex = extract_code(out, mbpp())
    assert ex.status == "code_block"
    assert "def add" in ex.code and "print" not in ex.code


def test_multiple_python_blocks_without_entry_point_takes_first() -> None:
    out = "```python\nx = 1\n```\n```python\ny = 2\n```"
    assert extract_code(out, mbpp()).code.strip() == "x = 1"


def test_untagged_fence_is_fallback() -> None:
    ex = extract_code("```\ndef add(a, b):\n    return a + b\n```", mbpp())
    assert ex.status == "fallback"
    assert run(ex.code, "add(2, 2)") == 4


def test_other_language_tag_is_fallback() -> None:
    ex = extract_code("```Python3\ndef add(a, b): return a+b\n```", mbpp())
    assert ex.status == "code_block"  # tag match is case-insensitive
    ex = extract_code("```text\ndef add(a, b): return a+b\n```", mbpp())
    assert ex.status == "fallback"


def test_missing_fences_raw_with_prose() -> None:
    out = (
        "Here is my answer.\ndef add(a, b):\n    return a + b\n\n"
        "This works because addition is commutative."
    )
    ex = extract_code(out, mbpp())
    assert ex.status == "raw"
    assert run(ex.code, "add(3, 4)") == 7
    assert "commutative" not in ex.code


def test_unterminated_block_truncated_generation() -> None:
    out = "```python\ndef add(a, b):\n    return a + b\n"  # hit max_new_tokens
    ex = extract_code(out, mbpp())
    assert ex.status == "code_block"
    assert run(ex.code, "add(1, 1)") == 2


def test_empty_outputs() -> None:
    assert extract_code("", mbpp()).status == "empty"
    assert extract_code("   \n", mbpp()).status == "empty"
    assert extract_code("```python\n```", mbpp()).status == "empty"


def test_strips_main_guard_asserts_prints_and_entry_point_calls() -> None:
    out = (
        "```python\nimport math\n\nCONST = 3\n\n"
        "def add(a, b):\n    return a + b\n\n"
        "assert add(1, 2) == 3\nprint(add(1, 2))\nresult = add(5, 5)\n"
        "for x in range(3):\n    print(add(x, x))\n\n"
        'if __name__ == "__main__":\n    import sys\n    print(add(int(sys.argv[1]), 2))\n```'
    )
    ex = extract_code(out, mbpp())
    assert "assert" not in ex.code and "print" not in ex.code and "__main__" not in ex.code
    assert "result =" not in ex.code
    assert "import math" in ex.code and "CONST = 3" in ex.code  # legit top-level code kept
    assert set(ex.notes) == {"stripped_main_guard", "stripped_asserts", "stripped_top_level_calls"}
    assert run(ex.code, "add(1, 2)") == 3


def test_main_guard_reversed_comparison_and_docstring_kept() -> None:
    code = '"""Module doc."""\ndef f():\n    return 1\nif "__main__" == __name__:\n    f()\n'
    cleaned, notes = strip_test_code(code, "f")
    assert "__main__" not in cleaned and '"""Module doc."""' in cleaned
    assert notes == ["stripped_main_guard"]


def test_syntax_error_code_is_cleaned_line_wise() -> None:
    out = (
        "```python\ndef add(a, b)\n    return a + b\n\nprint(add(1, 2))\n"
        "if __name__ == '__main__':\n    add(1, 2)\n```"
    )
    ex = extract_code(out, mbpp())
    assert "print" not in ex.code and "__main__" not in ex.code
    assert "def add(a, b)" in ex.code  # left for the sandbox to report as syntax_error


def test_humaneval_indented_body_only() -> None:
    out = (
        "```python\n    for i, a in enumerate(numbers):\n        for b in numbers[i + 1:]:\n"
        "            if abs(a - b) < threshold:\n                return True\n    return False\n```"
    )
    ex = extract_code(out, he())
    assert "reattached_signature" in ex.notes
    assert ex.code.startswith("from typing import List")
    assert run(ex.code, "has_close([1.0, 1.1], 0.5)") is True


def test_humaneval_unindented_body_only() -> None:
    out = (
        "```python\n"
        "return any(abs(a - b) < threshold for a in numbers for b in numbers if a is not b)\n```"
    )
    ex = extract_code(out, he())
    assert "reattached_signature" in ex.notes
    assert run(ex.code, "has_close([1.0, 3.0], 0.5)") is False


def test_humaneval_full_function_missing_import_gets_prompt_imports() -> None:
    out = (
        "```python\ndef has_close(numbers: List[float], threshold: float) -> bool:\n"
        "    return any(abs(a - b) < threshold for i, a in enumerate(numbers)"
        " for b in numbers[i + 1:])\n```"
    )
    ex = extract_code(out, he())
    assert ex.notes == ["added_prompt_imports"]
    assert run(ex.code, "has_close([1.0, 1.2], 0.5)") is True


def test_humaneval_import_already_present_not_duplicated() -> None:
    out = (
        "```python\nfrom typing import List\n\ndef has_close(numbers: List[float], t: float)"
        " -> bool:\n    return False\n```"
    )
    ex = extract_code(out, he())
    assert ex.code.count("from typing import List") == 1 and ex.notes == []


def test_humaneval_helper_from_prompt_is_prepended() -> None:
    out = (
        "```python\ndef make_palindrome(string: str) -> str:\n"
        "    for i in range(len(string)):\n"
        "        if is_palindrome(string[i:]):\n"
        "            return string + string[:i][::-1]\n    return string\n```"
    )
    ex = extract_code(out, he(HE_HELPER_PROMPT, "make_palindrome"))
    assert "added_prompt_helpers" in ex.notes
    assert run(ex.code, "make_palindrome('cat')") == "catac"


def test_humaneval_helper_redefined_by_model_not_duplicated() -> None:
    out = (
        "```python\ndef is_palindrome(s):\n    return s == s[::-1]\n\n"
        "def make_palindrome(string: str) -> str:\n    return string\n```"
    )
    ex = extract_code(out, he(HE_HELPER_PROMPT, "make_palindrome"))
    assert ex.code.count("def is_palindrome") == 1


def test_mbpp_never_reattaches_signature() -> None:
    ex = extract_code("```python\n    return a + b\n```", mbpp())
    assert "reattached_signature" not in ex.notes


def test_indented_fence_inside_list_is_dedented() -> None:
    out = "1. Code:\n   ```python\n   def add(a, b):\n       return a + b\n   ```\n"
    ex = extract_code(out, mbpp())
    assert ex.status == "code_block"
    assert run(ex.code, "add(1, 5)") == 6


@pytest.mark.parametrize("tag", ["py", "python", "PYTHON"])
def test_python_tag_variants(tag: str) -> None:
    ex = extract_code(f"```{tag}\ndef add(a, b):\n    return a + b\n```", mbpp())
    assert ex.status == "code_block"
