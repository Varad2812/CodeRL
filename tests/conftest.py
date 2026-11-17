"""Hand-written fixtures mimicking MBPP / HumanEval rows. No network access."""

from __future__ import annotations

from typing import Any

import pytest

from coderl.data.loaders import humaneval as he_mod
from coderl.data.loaders import mbpp as mbpp_mod

MBPP_ROWS: dict[str, list[dict[str, Any]]] = {
    "train": [
        {
            "task_id": 11,
            "text": "Write a python function to remove first and last occurrence of a given "
            "character from the string.",
            "code": "def remove_Occ(s,ch): \r\n    i = s.find(ch)\r\n    if i == -1:\r\n"
            "        return s\r\n    s = s[:i] + s[i + 1:]\r\n    j = s.rfind(ch)\r\n"
            "    if j != -1:\r\n        s = s[:j] + s[j + 1:]\r\n    return s ",
            "test_list": [
                'assert remove_Occ("hello","l") == "heo"',
                'assert remove_Occ("abcda","a") == "bcd"',
                'assert remove_Occ("PHP","P") == "H"',
            ],
            "test_setup_code": "",
            "challenge_test_list": ['assert remove_Occ("hellolloll","l") == "helollol"'],
        },
        {
            "task_id": 12,
            "text": "Write a function to sort a given matrix in ascending order according to "
            "the sum of its rows.",
            "code": "def sort_matrix(M):\r\n    return sorted(M, key=sum)",
            "test_list": [
                "assert sort_matrix([[1, 2, 3], [2, 4, 5], [1, 1, 1]])"
                "==[[1, 1, 1], [1, 2, 3], [2, 4, 5]]",
                "assert sort_matrix([[1, 2, 3], [-2, 4, -5], [1, -1, 1]])"
                "==[[-2, 4, -5], [1, -1, 1], [1, 2, 3]]",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 13,
            "text": "Write a function to count the most common words in a dictionary.",
            "code": "from collections import Counter\r\ndef count_common(words):\r\n"
            "    return Counter(words).most_common(4)",
            "test_list": [
                "assert count_common(['a','a','b']) == [('a', 2), ('b', 1)]",
                "assert count_common(['x']) == [('x', 1)]",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
    ],
    "validation": [
        {
            "task_id": 601,
            "text": "Write a function to compute the sum of squares of the first n numbers.",
            "code": "def square_sum(n):\r\n    return sum(i * i for i in range(1, n + 1))",
            "test_list": ["assert square_sum(3) == 14", "assert square_sum(1) == 1"],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
    ],
    "test": [
        {
            # lightly-edited near-duplicate of train task 11 (collision) + unrelated one
            "task_id": 801,
            "text": "Write a python function to remove first and last occurrence of a given "
            "character from the given string.",
            "code": "def remove_Occ(s,ch):\r\n    return s.replace(ch, '', 2)",
            "test_list": ['assert remove_Occ("hello","l") == "heo"'],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        {
            "task_id": 802,
            "text": "Write a function to reverse words separated by spaces in a given string.",
            "code": "def reverse_words(s):\r\n    return ' '.join(reversed(s.split()))",
            "test_list": ["assert reverse_words('a b c') == 'c b a'"],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
    ],
    "prompt": [],
}

HUMANEVAL_ROWS: list[dict[str, Any]] = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n    '''Add two numbers.'''\n",
        "canonical_solution": "    return a + b\n",
        "entry_point": "add",
        "test": "\n\nMETADATA = {'author': 'x'}\n\n\ndef check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
        "    assert candidate(-1, 1) == 0, 'neg'\n",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def double_all(xs):\n    '''Double each element.'''\n",
        "canonical_solution": "    return [2 * x for x in xs]\n",
        "entry_point": "double_all",
        "test": "def check(candidate):\n"
        "    import math\n"
        "    base = [1, 2, 3]\n"
        "    assert candidate(base) == [2, 4, 6]\n"
        "    for n in range(3):\n"
        "        assert candidate([n]) == [2 * n]\n",
    },
    {
        # no check() function at all -> must hit the fallback path
        "task_id": "HumanEval/2",
        "prompt": "def ident(x):\n    '''Identity.'''\n",
        "canonical_solution": "    return x\n",
        "entry_point": "ident",
        "test": "def check(candidate):\n    pass\n",
    },
]


@pytest.fixture
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mbpp_mod, "_load_hf", lambda split: MBPP_ROWS[split])
    monkeypatch.setattr(he_mod, "_load_hf", lambda split: HUMANEVAL_ROWS)
