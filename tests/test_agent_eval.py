from __future__ import annotations

from pathlib import Path

from evals.agent import live_runner, runner
from evals.agent.evaluators import evaluate_task
from evals.agent.report import compute_metrics, render_markdown
from evals.agent.semantics import FAILURE_STAGES, canonical_failure_stage, tool_event_summary
from src.planner.models import Plan, PlanStep, StepStatus


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


def test_failure_stage_taxonomy_is_canonicalized_in_metrics() -> None:
    records = [
        _metric_record("normal", False) | {"failure_stage": "router"},
        _metric_record("normal", False) | {"failure_stage": "runner"},
        _metric_record("normal", False) | {"failure_stage": "plan_validation"},
    ]

    metrics = compute_metrics(records)

    assert metrics["failure_stage_distribution"] == {
        "routing": 1,
        "runtime": 1,
        "plan_validation": 1,
    }
    assert canonical_failure_stage("planner") == "planning"
    assert set(FAILURE_STAGES) >= {"routing", "planning", "plan_validation", "runtime", "unknown"}


def test_report_renders_split_tool_proposal_metrics() -> None:
    metrics = compute_metrics(
        [
            _metric_record("normal", True)
            | {
                "tool_proposals": ["calculator", "read_text_file"],
                "model_tool_proposals": ["calculator"],
                "planned_tool_steps": ["read_text_file"],
            }
        ],
        metadata={"mode": "live_e2e"},
    )

    rendered = render_markdown(metrics)

    assert "Tool proposals (legacy assertions): 2" in rendered
    assert "Model/runtime tool proposals: 1" in rendered
    assert "Planned tool steps: 1" in rendered


def test_tool_event_summary_separates_model_plan_and_execution() -> None:
    summary = tool_event_summary(
        [
            {"tool": "calculator", "event": "model_tool_proposed", "source": "model"},
            {"tool": "read_text_file", "event": "planned_tool_step", "source": "plan"},
            {"tool": "read_text_file", "event": "execution_attempt"},
            {"tool": "write_text_file", "event": "policy_rejected", "source": "plan_policy"},
        ]
    )

    assert summary["model_tool_proposals"] == ["calculator"]
    assert summary["planned_tool_steps"] == ["read_text_file"]
    assert summary["actual_tool_executions"] == ["read_text_file"]
    assert summary["tool_proposals"] == ["calculator", "read_text_file"]
    assert summary["tool_policy_rejections"] == 1


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


def test_live_dataset_schema_and_coverage() -> None:
    examples = live_runner.load_dataset()

    assert 30 <= len(examples) <= 40
    assert len({example["id"] for example in examples}) == len(examples)
    assert examples[0]["id"] == "live_001"

    categories = {example["category"] for example in examples}
    assert {
        "single_tool_calculator",
        "planned_file_task",
        "ambiguous_request",
        "failure_boundary_invalid_tool_args",
        "policy_planned_write_rejected",
        "memory_retrieval",
        "router_runtime_contract_conflict_web",
        "planner_failure_missing_input_file",
        "model_protocol_failure_runtime",
    } <= categories

    valid_routes = {"direct_answer", "single_tool", "web_lookup", "planned_task", "deep_research", "clarification"}
    for example in examples:
        assert set(example) >= {"id", "suite", "category", "input", "expected_route", "expected"}
        assert example["expected_route"] in valid_routes
        assert example["suite"] in {"normal", "regression"}
        assert isinstance(example["expected"], dict)


def test_planned_record_uses_planned_steps_not_model_proposals() -> None:
    plan = Plan(
        goal="save",
        steps=[
            PlanStep(
                id="step_1",
                description="read",
                tool_name="read_text_file",
                arguments={"path": "notes.txt"},
                status=StepStatus.COMPLETED,
                actual_output="notes",
            )
        ],
    )

    deterministic = runner._plan_record(plan, "done", "completed")
    live = live_runner._plan_record(plan, "done", "completed", [])

    for record in (deterministic, live):
        assert record["model_tool_proposals"] == []
        assert record["planned_tool_steps"] == ["read_text_file"]
        assert record["tool_proposals"] == ["read_text_file"]
        assert record["tool_execution_attempts_by_name"] == ["read_text_file"]


def test_runtime_record_uses_model_proposals_not_planned_steps() -> None:
    class DummyLLM:
        emitted_tool_calls = ["calculator"]

    class DummyRegistry:
        observed_tool_events = [
            {"tool": "calculator", "event": "execution_attempt"},
            {"tool": "calculator", "event": "execution_success"},
        ]

    record = runner._runtime_record("4", "final_answer", DummyRegistry(), DummyLLM())

    assert record["model_tool_proposals"] == ["calculator"]
    assert record["planned_tool_steps"] == []
    assert record["actual_tool_executions"] == ["calculator"]


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
