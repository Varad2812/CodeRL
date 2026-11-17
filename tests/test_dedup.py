from coderl.data.dedup import check_contamination, normalize, remove_contaminated, shingles
from coderl.data.schema import Problem


def _p(tid: str, prompt: str, sol: str | None = None) -> Problem:
    return Problem(
        task_id=tid,
        source="mbpp",
        split="train",
        prompt=prompt,
        entry_point="f",
        tests=["assert True"],
        canonical_solution=sol,
    )


BASE = (
    "Write a python function to remove the first and last occurrence of a given "
    "character from the string and return the resulting string."
)


def test_normalize_and_shingles() -> None:
    assert normalize('"""Hello,  World!"""') == normalize("hello world")
    assert shingles("a b c") == {"a b c"}
    assert len(shingles("one two three four five six")) == 2


def test_exact_prompt_and_solution() -> None:
    train = [_p("t/1", BASE, "def f(x):\n    return x + 1"), _p("t/2", "unrelated stuff here")]
    ev = [_p("e/1", BASE.upper() + "  ", None), _p("e/2", "zzz", "def f(x): return x+1")]
    rep = check_contamination(train, ev)
    kinds = {(p.train_id, p.eval_id, p.kind) for p in rep.pairs}
    assert ("t/1", "e/1", "exact_prompt") in kinds
    assert ("t/1", "e/2", "exact_solution") in kinds
    assert rep.flagged_train_ids == ["t/1"]
    assert [p.task_id for p in remove_contaminated(train, rep)] == ["t/2"]


def test_near_duplicate_detected_and_unrelated_not() -> None:
    edited = BASE + " Thanks"  # light edit: Jaccard ~0.95
    train = [_p("t/1", BASE), _p("t/2", "Compute the area of a circle given its radius as a float")]
    ev = [_p("e/1", edited)]
    rep = check_contamination(train, ev)
    assert [(p.train_id, p.kind) for p in rep.pairs] == [("t/1", "near_prompt")]
    assert 0.8 <= rep.pairs[0].similarity < 1.0
    assert rep.n_near == 1 and rep.n_exact == 0


def test_no_contamination() -> None:
    rep = check_contamination([_p("t/1", BASE)], [_p("e/1", "totally different words go here ok")])
    assert rep.n_train_flagged == 0 and rep.pairs == []
