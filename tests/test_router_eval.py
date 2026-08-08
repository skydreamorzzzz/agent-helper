from __future__ import annotations

from pathlib import Path

import pytest

from evals.router.runner import (
    RouterExample,
    build_route_fn,
    compute_metrics,
    evaluate_examples,
    filter_examples_by_split,
    load_dataset,
    _route_embedding_cascade,
)
from src.planner.models import Route, RouteDecision
from src.planner.router import RouteCandidate


def test_router_dataset_loads_with_required_route_coverage_and_splits() -> None:
    examples = load_dataset()
    expected_routes = {str(example.expected_route) for example in examples}

    assert len(examples) == 60
    assert expected_routes == {str(route.value) for route in Route}
    assert {example.split for example in examples} == {"dev", "test"}
    for split in ("dev", "test"):
        split_routes = {str(example.expected_route) for example in examples if example.split == split}
        assert split_routes == {str(route.value) for route in Route}


def test_router_dataset_rejects_invalid_route_label(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id":"bad","input":"hello","expected_route":"not_a_route","category":"x","language":"en"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid router dataset row"):
        load_dataset(dataset)


def test_router_dataset_rejects_invalid_split(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id":"bad","input":"hello","expected_route":"direct_answer","category":"x","language":"en","split":"train"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid router dataset row"):
        load_dataset(dataset)


def test_router_split_filtering() -> None:
    examples = load_dataset()
    dev = filter_examples_by_split(examples, "dev")
    test = filter_examples_by_split(examples, "test")
    all_examples = filter_examples_by_split(examples, "all")

    assert len(dev) == 42
    assert len(test) == 18
    assert len(all_examples) == 60
    assert all(example.split == "dev" for example in dev)
    assert all(example.split == "test" for example in test)


def test_router_metrics_accuracy_confusion_and_macro_f1() -> None:
    records = [
        {
            "expected_route": "direct_answer",
            "predicted_route": "direct_answer",
            "correct": True,
            "category": "stable",
            "llm_escalated": True,
        },
        {
            "expected_route": "web_lookup",
            "predicted_route": "direct_answer",
            "correct": False,
            "category": "current",
            "llm_escalated": True,
        },
        {
            "expected_route": "web_lookup",
            "predicted_route": "web_lookup",
            "correct": True,
            "category": "current",
            "llm_escalated": False,
        },
    ]

    metrics = compute_metrics(records, mode="unit", split="test", llm_call_count=5)

    assert metrics["dataset_version"] == "router-v1"
    assert metrics["split"] == "test"
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"]["web_lookup"]["direct_answer"] == 1
    assert metrics["confusion_matrix"]["web_lookup"]["web_lookup"] == 1
    assert metrics["per_route"]["web_lookup"]["precision"] == pytest.approx(1.0)
    assert metrics["per_route"]["web_lookup"]["recall"] == pytest.approx(0.5)
    assert metrics["per_route"]["web_lookup"]["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0 + 0 + 0 + 0 + 2 / 3) / 6)
    assert metrics["category_accuracy"]["current"]["accuracy"] == pytest.approx(0.5)
    assert metrics["llm_call_count"] == 5
    assert metrics["llm_escalated_examples"] == 2
    assert metrics["llm_escalation_rate"] == pytest.approx(2 / 3)


def test_llm_escalation_rate_counts_examples_not_total_calls() -> None:
    records = [
        {
            "expected_route": "direct_answer",
            "predicted_route": "direct_answer",
            "correct": True,
            "category": "stable",
            "llm_escalated": True,
        },
        {
            "expected_route": "web_lookup",
            "predicted_route": "web_lookup",
            "correct": True,
            "category": "current",
            "llm_escalated": False,
        },
    ]

    metrics = compute_metrics(records, mode="unit", split="test", llm_call_count=3)

    assert metrics["llm_call_count"] == 3
    assert metrics["llm_escalated_examples"] == 1
    assert metrics["llm_escalation_rate"] == pytest.approx(0.5)


def test_router_eval_runs_fake_router_and_counts_llm_calls() -> None:
    examples = [
        RouterExample(
            id="a",
            input="hello",
            expected_route=Route.DIRECT_ANSWER,
            category="casual",
            language="en",
            split="dev",
        ),
        RouterExample(
            id="b",
            input="price",
            expected_route=Route.WEB_LOOKUP,
            category="current",
            language="en",
            split="test",
        ),
    ]
    calls = 0

    def fake_router(user_input: str) -> RouteDecision:
        nonlocal calls
        calls += 1
        route = Route.WEB_LOOKUP if "price" in user_input else Route.DIRECT_ANSWER
        return RouteDecision(route=route, reason=f"fake {route}")

    records, metrics = evaluate_examples(
        examples,
        fake_router,
        mode="fake",
        split="all",
        llm_call_count=lambda: calls,
    )

    assert [record["correct"] for record in records] == [True, True]
    assert metrics["accuracy"] == 1.0
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_escalated_examples"] == 2


def test_constraint_only_mode_does_not_instantiate_request_router(monkeypatch) -> None:
    import evals.router.runner as runner

    monkeypatch.setattr(runner, "RequestRouter", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")))

    route_fn, counter, metadata = build_route_fn("constraint_only")

    assert route_fn("hello").route == Route.DIRECT_ANSWER
    assert counter() == 0
    assert metadata["embedding_model"] == ""


def test_lexical_baseline_does_not_call_llm() -> None:
    route_fn, counter, metadata = build_route_fn("lexical_baseline")

    assert route_fn("What is the latest Tavily API pricing?").route == Route.WEB_LOOKUP
    assert counter() == 0
    assert metadata["llm_model"] == ""


def test_embedding_only_does_not_call_llm() -> None:
    route_fn, counter, metadata = build_route_fn(
        "embedding_only",
        similarity_threshold=0.0,
        margin_threshold=0.0,
    )

    route_fn("Summarize and save it somewhere")

    assert counter() == 0
    assert metadata["embedding_model"]


class AlwaysUncertainEmbeddingRouter:
    def route(self, user_input: str):
        return None


class KnownEmbeddingRouter:
    def route(self, user_input: str):
        if "known" not in user_input:
            return None
        return RouteCandidate(
            RouteDecision(route=Route.CLARIFICATION, reason="known embedding"),
            confidence=1.0,
            source="embedding",
        )


class FakeLLMRouter:
    def __init__(self) -> None:
        self.calls = 0

    def _route_with_llm(self, user_input: str, *, memory_context: str) -> RouteDecision:
        self.calls += 1
        return RouteDecision(route=Route.DIRECT_ANSWER, reason="llm fallback")


class EmptyConstraintRouter:
    def route(self, user_input: str):
        return None


def test_embedding_hybrid_calls_llm_only_when_embedding_is_uncertain() -> None:
    llm = FakeLLMRouter()
    constraint = EmptyConstraintRouter()

    known = _route_embedding_cascade(
        "known request",
        constraint_router=constraint,
        embedding_router=KnownEmbeddingRouter(),
        llm_router=llm,
    )
    unknown = _route_embedding_cascade(
        "unknown request",
        constraint_router=constraint,
        embedding_router=AlwaysUncertainEmbeddingRouter(),
        llm_router=llm,
    )

    assert known.route == Route.CLARIFICATION
    assert unknown.route == Route.DIRECT_ANSWER
    assert llm.calls == 1
