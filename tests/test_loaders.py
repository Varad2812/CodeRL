import pytest

from coderl.data.loaders import load_source
from coderl.data.loaders.humaneval import split_check_function
from coderl.data.loaders.mbpp import parse_entry_point


def test_mbpp_mapping(fake_datasets: None) -> None:
    probs = {p.task_id: p for p in load_source("mbpp", "train")}
    p = probs["mbpp/11"]
    assert p.source == "mbpp" and p.split == "train"
    assert p.entry_point == "remove_Occ"
    assert 'assert remove_Occ("hello","l") == "heo"' in p.prompt  # first test in prompt
    assert "\r" not in p.canonical_solution  # type: ignore[operator]
    # challenge test appended and flagged
    assert len(p.tests) == 4
    assert p.metadata["n_challenge_tests"] == 1
    assert p.metadata["challenge_test_indices"] == [3]
    assert probs["mbpp/13"].entry_point == "count_common"  # not the Counter import


def test_mbpp_entry_point_prefers_function_called_by_test() -> None:
    code = "def helper(x):\n    return x\n\ndef main_fn(x):\n    return helper(x)"
    assert parse_entry_point(code, "assert main_fn(1) == 1") == "main_fn"


def test_mbpp_bad_split(fake_datasets: None) -> None:
    with pytest.raises(ValueError):
        load_source("mbpp", "bogus")


def test_humaneval_split_and_rename(fake_datasets: None) -> None:
    probs = {p.task_id: p for p in load_source("humaneval", "test")}
    p = probs["humaneval/0"]
    assert p.metadata["test_split"] == "split"
    assert p.tests == [
        "assert add(1, 2) == 3",
        "assert add(0, 0) == 0",
        "assert add(-1, 1) == 0, 'neg'",
    ]
    assert all("candidate" not in t for t in p.tests)
    assert "METADATA" in p.test_setup


def test_humaneval_context_and_imports(fake_datasets: None) -> None:
    p = {p.task_id: p for p in load_source("humaneval", "test")}["humaneval/1"]
    assert p.metadata["test_split"] == "split"
    assert "import math" in p.test_setup  # imports hoisted to setup
    assert len(p.tests) == 2
    assert p.tests[0].startswith("base = [1, 2, 3]")  # dependent assignment stays with assert
    assert "for n in range(3)" in p.tests[1] and "base = [1, 2, 3]" in p.tests[1]


def test_humaneval_fallback(fake_datasets: None) -> None:
    p = {p.task_id: p for p in load_source("humaneval", "test")}["humaneval/2"]
    assert p.metadata["test_split"] == "fallback"
    assert len(p.tests) == 1
    assert p.tests[0].rstrip().endswith("check(ident)")


def test_split_errors() -> None:
    with pytest.raises(ValueError):
        split_check_function("x = (", "f")
    with pytest.raises(ValueError):
        split_check_function("y = 1", "f")


def test_apps_stub_and_registry() -> None:
    with pytest.raises(NotImplementedError):
        load_source("apps", "train")
    with pytest.raises(ValueError):
        load_source("nope", "train")
