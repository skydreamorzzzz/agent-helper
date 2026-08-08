from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import load_settings
from src.llm.client import LocalLLMClient
from src.planner.embedding_router import EmbeddingRouter, HashingEmbedder
from src.planner.models import Route, RouteDecision
from src.planner.router import ConstraintRouter, LLMRouter, RequestRouter

from evals.router.report import write_predictions, write_report

DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
RESULTS_ROOT = Path(__file__).with_name("results")
DATASET_VERSION = "router-v1"
DEFAULT_EMBEDDING_MODEL = os.getenv("ROUTER_EMBEDDING_MODEL", "hashing-multilingual-v1")
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("ROUTER_EMBEDDING_SIMILARITY_THRESHOLD", "0.32"))
DEFAULT_MARGIN_THRESHOLD = float(os.getenv("ROUTER_EMBEDDING_MARGIN_THRESHOLD", "0.04"))
BenchmarkMode = Literal[
    "constraint_only",
    "lexical_baseline",
    "embedding_only",
    "current_hybrid",
    "embedding_hybrid",
]
BenchmarkSplit = Literal["dev", "test", "all"]


class RouterExample(BaseModel):
    id: str = Field(min_length=1)
    input: str = Field(min_length=1)
    expected_route: Route
    category: str = Field(min_length=1)
    language: str = Field(min_length=1)
    note: str = ""
    split: str

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in {"en", "zh", "mixed", "ood"}:
            raise ValueError("language must be one of: en, zh, mixed, ood")
        return value

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str) -> str:
        if value not in {"dev", "test"}:
            raise ValueError("split must be one of: dev, test")
        return value


@dataclass
class CountingLLM:
    inner: LocalLLMClient
    count: int = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.count += 1
        return self.inner.chat(messages)


RouteFn = Callable[[str], RouteDecision]


def load_dataset(path: Path = DATASET_PATH) -> list[RouterExample]:
    examples: list[RouterExample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                example = RouterExample.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(f"Invalid router dataset row {line_no}: {exc}") from exc
            if example.id in seen_ids:
                raise ValueError(f"Duplicate router dataset id at row {line_no}: {example.id}")
            seen_ids.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError(f"Router dataset is empty: {path}")
    return examples


def filter_examples_by_split(examples: list[RouterExample], split: BenchmarkSplit) -> list[RouterExample]:
    if split == "all":
        return examples
    return [example for example in examples if example.split == split]


def evaluate_examples(
    examples: list[RouterExample],
    route_fn: RouteFn,
    *,
    mode: str,
    split: str = "all",
    llm_call_count: Callable[[], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        calls_before = llm_call_count() if llm_call_count is not None else 0
        try:
            decision = route_fn(example.input)
            predicted = str(decision.route)
            reason = decision.reason
        except Exception as exc:
            predicted = "error"
            reason = f"{type(exc).__name__}: {exc}"
        calls_after = llm_call_count() if llm_call_count is not None else calls_before
        llm_calls = max(0, calls_after - calls_before)
        records.append(
            {
                "id": example.id,
                "input": example.input,
                "expected_route": str(example.expected_route),
                "predicted_route": predicted,
                "correct": predicted == str(example.expected_route),
                "reason": reason,
                "category": example.category,
                "language": example.language,
                "note": example.note,
                "split": example.split,
                "llm_calls": llm_calls,
                "llm_escalated": llm_calls > 0,
            }
        )

    llm_calls = llm_call_count() if llm_call_count is not None else 0
    return records, compute_metrics(records, mode=mode, split=split, llm_call_count=llm_calls)


def compute_metrics(
    records: list[dict[str, Any]],
    *,
    mode: str,
    split: str = "all",
    llm_call_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels = sorted({str(route.value) for route in Route} | {str(record["predicted_route"]) for record in records})
    total = len(records)
    correct = sum(1 for record in records if record["correct"])
    accuracy = correct / total if total else 0.0

    confusion: dict[str, dict[str, int]] = {label: {predicted: 0 for predicted in labels} for label in labels}
    for record in records:
        expected = str(record["expected_route"])
        predicted = str(record["predicted_route"])
        confusion.setdefault(expected, {label: 0 for label in labels})
        if predicted not in confusion[expected]:
            confusion[expected][predicted] = 0
        confusion[expected][predicted] += 1

    per_route: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(expected, {}).get(label, 0) for expected in labels if expected != label)
        fn = sum(count for predicted, count in confusion.get(label, {}).items() if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        support = sum(confusion.get(label, {}).values())
        per_route[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    route_labels = [str(route.value) for route in Route]
    macro_f1 = sum(float(per_route[label]["f1"]) for label in route_labels) / len(route_labels)

    category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for record in records:
        item = category_counts[str(record["category"])]
        item["total"] += 1
        if record["correct"]:
            item["correct"] += 1
    category_accuracy = {
        category: {
            "correct": item["correct"],
            "total": item["total"],
            "accuracy": item["correct"] / item["total"] if item["total"] else 0.0,
        }
        for category, item in category_counts.items()
    }

    return {
        "dataset_version": DATASET_VERSION,
        "mode": mode,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "labels": labels,
        "per_route": per_route,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "category_accuracy": category_accuracy,
        "llm_call_count": llm_call_count,
        "llm_escalated_examples": sum(1 for record in records if record.get("llm_escalated")),
        "llm_escalation_rate": (
            sum(1 for record in records if record.get("llm_escalated")) / total if total else 0.0
        ),
        "metadata": metadata or {},
    }


def build_route_fn(
    mode: BenchmarkMode,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> tuple[RouteFn, Callable[[], int], dict[str, Any]]:
    base_metadata = {
        "git_commit": _git_commit(),
        "embedding_model": "",
        "llm_model": "",
        "similarity_threshold": None,
        "margin_threshold": None,
    }
    if mode == "constraint_only":
        constraint = ConstraintRouter()

        def route_with_constraints(user_input: str) -> RouteDecision:
            candidate = constraint.route(user_input)
            if candidate is not None:
                return candidate.decision
            return RouteDecision(route=Route.DIRECT_ANSWER, reason="constraint fallback: direct_answer")

        return route_with_constraints, lambda: 0, base_metadata

    if mode == "lexical_baseline":
        router = RequestRouter()
        return lambda user_input: router.route(user_input), lambda: 0, base_metadata

    if mode == "embedding_only":
        constraint = ConstraintRouter()
        embedding_router = EmbeddingRouter(
            embedder=HashingEmbedder(model_name=embedding_model),
            similarity_threshold=similarity_threshold,
            margin_threshold=margin_threshold,
        )
        metadata = {
            **base_metadata,
            "embedding_model": embedding_router.model_name,
            "similarity_threshold": similarity_threshold,
            "margin_threshold": margin_threshold,
        }

        def route_with_embedding(user_input: str) -> RouteDecision:
            return _route_embedding_cascade(
                user_input,
                constraint_router=constraint,
                embedding_router=embedding_router,
                llm_router=None,
            )

        return route_with_embedding, lambda: 0, metadata

    settings = load_settings()
    counting = CountingLLM(
        LocalLLMClient(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
            model=settings.local_llm_model,
            timeout=settings.local_llm_timeout,
        )
    )
    if mode == "current_hybrid":
        router = RequestRouter(counting)
        return lambda user_input: router.route(user_input), lambda: counting.count, {
            **base_metadata,
            "llm_model": settings.local_llm_model,
        }

    constraint = ConstraintRouter()
    embedding_router = EmbeddingRouter(
        embedder=HashingEmbedder(model_name=embedding_model),
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
    )
    llm_router = LLMRouter(counting)
    metadata = {
        **base_metadata,
        "embedding_model": embedding_router.model_name,
        "llm_model": settings.local_llm_model,
        "similarity_threshold": similarity_threshold,
        "margin_threshold": margin_threshold,
    }

    def route_with_embedding_hybrid(user_input: str) -> RouteDecision:
        return _route_embedding_cascade(
            user_input,
            constraint_router=constraint,
            embedding_router=embedding_router,
            llm_router=llm_router,
        )

    return route_with_embedding_hybrid, lambda: counting.count, metadata


def _route_embedding_cascade(
    user_input: str,
    *,
    constraint_router: ConstraintRouter,
    embedding_router: EmbeddingRouter,
    llm_router: LLMRouter | None,
) -> RouteDecision:
    constraint = constraint_router.route(user_input)
    if constraint and constraint.final:
        return constraint.decision

    embedding = embedding_router.route(user_input)
    if embedding is not None:
        return embedding.decision

    if llm_router is not None:
        llm_decision = llm_router._route_with_llm(user_input, memory_context="")
        if llm_decision is not None:
            if (
                constraint is not None
                and constraint.decision.route == Route.WEB_LOOKUP
                and llm_decision.route == Route.DIRECT_ANSWER
            ):
                return constraint.decision
            return llm_decision

    if constraint is not None:
        return constraint.decision
    return RouteDecision(route=Route.DIRECT_ANSWER, reason="embedding cascade fallback: direct_answer")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def run_benchmark(
    *,
    mode: BenchmarkMode,
    split: BenchmarkSplit = "test",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    dataset_path: Path = DATASET_PATH,
    out_root: Path = RESULTS_ROOT,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    examples = filter_examples_by_split(load_dataset(dataset_path), split)
    route_fn, counter, metadata = build_route_fn(
        mode,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
    )
    records, metrics = evaluate_examples(examples, route_fn, mode=mode, split=split, llm_call_count=counter)
    metrics["metadata"] = metadata

    out_dir = out_root / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(records, out_dir)
    write_report(metrics, records, out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir, records, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run router evaluation benchmark.")
    parser.add_argument(
        "--mode",
        choices=["constraint_only", "lexical_baseline", "embedding_only", "current_hybrid", "embedding_hybrid"],
        default="lexical_baseline",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    parser.add_argument("--margin-threshold", type=float, default=DEFAULT_MARGIN_THRESHOLD)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="path to router dataset jsonl")
    parser.add_argument("--out", default=str(RESULTS_ROOT), help="output root directory")
    args = parser.parse_args()

    out_dir, records, metrics = run_benchmark(
        mode=args.mode,
        split=args.split,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        margin_threshold=args.margin_threshold,
        dataset_path=Path(args.dataset),
        out_root=Path(args.out),
    )
    for record in records:
        print(
            f"{record['id']}\tcorrect={record['correct']}\t"
            f"expected={record['expected_route']}\tpredicted={record['predicted_route']}\t"
            f"reason={record['reason']}"
        )
        print(f"  input={record['input']}")
    print(f"Total accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(
        f"LLM calls: {metrics['llm_call_count']} "
        f"(escalated examples: {metrics['llm_escalated_examples']}, "
        f"rate={metrics['llm_escalation_rate']:.3f})"
    )
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
