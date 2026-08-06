from __future__ import annotations

from evals.gaia import is_correct, normalize_answer, select_questions
from evals.runner import _read_runtime_trajectory

import evals.runner as runner_module


def test_normalize_answer_strips_case_punct_and_articles() -> None:
    assert normalize_answer(" The United States! ") == "united states"
    assert normalize_answer("A dog") == "dog"
    assert normalize_answer("An apple") == "apple"


def test_normalize_answer_keeps_non_ascii() -> None:
    assert normalize_answer("Café") == "cafe"


def test_is_correct_numeric_with_currency_and_commas() -> None:
    assert is_correct("$1,234", "1234")
    assert is_correct("50%", "50")
    assert is_correct("5.25", "5.25")


def test_is_correct_list_order_insensitive() -> None:
    assert is_correct("3, 1, 2", "1,2,3")
    assert is_correct("apple;banana", "banana, apple")


def test_is_correct_exact_and_article() -> None:
    assert is_correct("dog", "a dog")
    assert not is_correct("cat", "dog")


def test_is_correct_empty_or_none_fails() -> None:
    assert not is_correct("", "dog")
    assert not is_correct("None", "")


def test_select_questions_filters_attachments_and_empties() -> None:
    metadata = {
        1: [
            {"task_id": "t1", "question": "q1?", "answer": "a", "level": 1, "file_name": ""},
            {"task_id": "t2", "question": "q2?", "answer": "a", "level": 1, "file_name": "file.pdf"},
            {"task_id": "t3", "question": "q3?", "answer": "a", "level": 1},
            {"task_id": "t4", "question": "q4?", "answer": "", "level": 1},
        ]
    }

    questions = select_questions(metadata, levels=(1,))

    assert [q.task_id for q in questions] == ["t1", "t3"]


def test_select_questions_only_keeps_requested_levels() -> None:
    metadata = {
        1: [{"task_id": "t1", "question": "q1?", "answer": "a", "level": 1}],
        2: [{"task_id": "t2", "question": "q2?", "answer": "a", "level": 2}],
    }

    questions = select_questions(metadata, levels=(1, 2))

    assert {q.level for q in questions} == {1, 2}


def test_select_questions_sampling_is_reproducible() -> None:
    metadata = {1: [{"task_id": f"t{i}", "question": f"q{i}?", "answer": "a", "level": 1} for i in range(10)]}

    first = select_questions(metadata, levels=(1,), limit=5, seed=42)
    second = select_questions(metadata, levels=(1,), limit=5, seed=42)

    assert len(first) == 5
    assert [q.task_id for q in first] == [q.task_id for q in second]


def test_read_runtime_trajectory_handles_timestamped_logs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner_module, "LOGS_DIR", tmp_path)
    (tmp_path / "run-1.log").write_text(
        '2026-08-06 12:00:00 INFO {"event":"tool_call","tool":"search_web","arguments":{"query":"hello"}}\n',
        encoding="utf-8",
    )

    tools, queries = _read_runtime_trajectory("run-1")

    assert tools == ["search_web"]
    assert queries == ["hello"]
