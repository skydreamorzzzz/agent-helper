from __future__ import annotations

from pathlib import Path

from evals.agent.evaluators import evaluate_task
from evals.agent.report import compute_metrics


def test_route_mismatch_but_task_succeeds(tmp_path: Path) -> None:
    (tmp_path / "out.md").write_text("done", encoding="utf-8")
    example = {
        "id": "case",
        "suite": "normal",
        "expected_route": "single_tool",
        "expected": {
            "final_status": "completed",
            "output_file": "out.md",
            "file_should_exist": True,
            "file_contains": ["done"],
        },
    }
    record = {
        "actual_route": "planned_task",
        "final_status": "completed",
        "stopped_reason": "completed",
        "final_answer": "done",
        "tool_proposals": ["write_text_file"],
        "tool_execution_attempts_by_name": ["write_text_file"],
        "tool_execution_failures": 0,
    }

    result = evaluate_task(example, record, tmp_path)

    assert result.task_success is True
    assert result.route_correct is False
    assert result.failure_stage == ""


def test_permission_rejection_is_safety_success(tmp_path: Path) -> None:
    (tmp_path / "existing.md").write_text("keep me", encoding="utf-8")
    example = {
        "id": "case",
        "suite": "normal",
        "expected_route": "planned_task",
        "expected": {
            "permission_outcome": "rejected",
            "side_effect_should_happen": False,
            "output_file": "existing.md",
            "file_should_exist": True,
            "file_contains": ["keep me"],
            "file_not_contains": ["new content"],
            "tool_proposals": ["write_text_file"],
            "tool_executions_not": ["write_text_file"],
        },
    }
    record = {
        "actual_route": "planned_task",
        "final_status": "confirmation_required",
        "stopped_reason": "confirmation_required",
        "final_answer": "needs confirmation",
        "tool_proposals": ["write_text_file"],
        "tool_execution_attempts_by_name": [],
        "tool_policy_rejections": 1,
        "tool_execution_failures": 0,
    }

    result = evaluate_task(example, record, tmp_path)

    assert result.task_success is True
    assert result.route_correct is True
    assert result.failure_stage == ""


def test_tool_proposal_policy_rejection_is_not_execution_failure() -> None:
    records = [
        {
            "suite": "normal",
            "task_success": True,
            "route_correct": True,
            "tool_proposals": ["write_text_file"],
            "tool_execution_attempts": 0,
            "tool_execution_successes": 0,
            "tool_execution_failures": 0,
            "tool_policy_rejections": 1,
            "llm_calls": 1,
            "retry_count": 0,
            "replan_count": 0,
            "latency_ms": 1,
            "actual_route": "single_tool",
        }
    ]

    metrics = compute_metrics(records)

    assert metrics["tool_proposals"] == 1
    assert metrics["tool_execution_attempts"] == 0
    assert metrics["tool_execution_success_rate"] == 1.0
    assert metrics["tool_policy_rejections"] == 1


def test_tool_execution_failure_counts_as_execution_failure() -> None:
    records = [
        {
            "suite": "normal",
            "task_success": False,
            "route_correct": True,
            "failure_stage": "tool_execution",
            "tool_proposals": ["read_text_file"],
            "tool_execution_attempts": 1,
            "tool_execution_successes": 0,
            "tool_execution_failures": 1,
            "tool_policy_rejections": 0,
            "llm_calls": 1,
            "retry_count": 0,
            "replan_count": 0,
            "latency_ms": 1,
            "actual_route": "single_tool",
        }
    ]

    metrics = compute_metrics(records)

    assert metrics["tool_execution_attempts"] == 1
    assert metrics["tool_execution_failures"] == 1
    assert metrics["tool_execution_success_rate"] == 0.0


def test_normal_and_regression_suite_metrics_are_separate() -> None:
    records = [
        _metric_record("normal", True),
        _metric_record("normal", False),
        _metric_record("regression", True),
    ]

    metrics = compute_metrics(records)

    assert metrics["normal_task_success_rate"] == 0.5
    assert metrics["regression_case_pass_rate"] == 1.0
    assert metrics["overall_integration_pass_rate"] == 2 / 3


def test_failure_attribution_only_for_failed_contract(tmp_path: Path) -> None:
    success = evaluate_task(
        {"expected_route": "direct_answer", "expected": {"final_status": "final_answer"}},
        {
            "actual_route": "direct_answer",
            "final_status": "final_answer",
            "stopped_reason": "final_answer",
            "final_answer": "ok",
            "tool_proposals": [],
            "tool_execution_failures": 0,
        },
        tmp_path,
    )
    failure = evaluate_task(
        {"expected_route": "direct_answer", "category": "plain", "expected": {"final_status": "final_answer"}},
        {
            "actual_route": "direct_answer",
            "final_status": "invalid_json_repair_failed",
            "stopped_reason": "invalid_json_repair_failed",
            "final_answer": "",
            "tool_proposals": [],
            "tool_execution_failures": 0,
        },
        tmp_path,
    )

    assert success.task_success is True
    assert success.failure_stage == ""
    assert failure.task_success is False
    assert failure.failure_stage == "recovery"


def _metric_record(suite: str, success: bool) -> dict:
    return {
        "suite": suite,
        "task_success": success,
        "route_correct": True,
        "failure_stage": "" if success else "final_answer",
        "tool_proposals": [],
        "tool_execution_attempts": 0,
        "tool_execution_successes": 0,
        "tool_execution_failures": 0,
        "tool_policy_rejections": 0,
        "llm_calls": 1,
        "retry_count": 0,
        "replan_count": 0,
        "latency_ms": 1,
        "actual_route": "direct_answer",
    }
