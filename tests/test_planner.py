from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from src.planner.executor import PlanExecutor
from src.planner.models import Plan, PlanStatus, PlanStep, RiskLevel, Route, StepStatus
from src.planner.repository import PlanRepository
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.base import Tool
from src.tools.calculator import CalculatorTool
from src.tools.file_tools import ReadTextFileTool, WriteTextFileTool
from src.tools.registry import ToolRegistry


class EchoArguments(BaseModel):
    value: str


class EchoTool(Tool):
    name = "echo"
    description = "Echo text"
    argument_schema = EchoArguments
    risk_level = RiskLevel.READ_ONLY

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def execute(self, arguments: EchoArguments) -> str:
        self.calls.append(arguments.value)
        return arguments.value


class FailingTool(Tool):
    name = "fail"
    description = "Always fail"
    argument_schema = EchoArguments
    risk_level = RiskLevel.READ_ONLY

    def execute(self, arguments: EchoArguments) -> str:
        raise RuntimeError("boom")


class DestructiveTool(Tool):
    name = "destroy"
    description = "Dangerous operation"
    argument_schema = EchoArguments
    risk_level = RiskLevel.DESTRUCTIVE

    def execute(self, arguments: EchoArguments) -> str:
        return "destroyed"


def make_registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def make_repo(tmp_path: Path) -> PlanRepository:
    return PlanRepository(tmp_path / "plans.sqlite3")


def test_simple_chat_routes_direct_answer() -> None:
    decision = RequestRouter().route("你好，随便聊聊")

    assert decision.route == Route.DIRECT_ANSWER


def test_single_tool_request_routes_single_tool() -> None:
    decision = RequestRouter().route("计算 123 * 456")

    assert decision.route == Route.SINGLE_TOOL


def test_multi_step_request_routes_planned_task() -> None:
    decision = RequestRouter().route("读取 todo.txt，整理三项任务，并保存到 today_plan.md")

    assert decision.route == Route.PLANNED_TASK


def test_missing_information_routes_clarification() -> None:
    decision = RequestRouter().route("帮我读取文件")

    assert decision.route == Route.CLARIFICATION
    assert decision.missing_information


def test_unknown_tool_fails_validation() -> None:
    plan = Plan(goal="bad", steps=[PlanStep(id="s1", description="bad", tool_name="missing")])

    result = PlanValidator(make_registry()).validate(plan)

    assert not result.ok
    assert "Unknown tool" in result.errors[0]


def test_invalid_arguments_fail_validation() -> None:
    plan = Plan(
        goal="bad args",
        steps=[PlanStep(id="s1", description="calc", tool_name="calculator", arguments={})],
    )

    result = PlanValidator(make_registry(CalculatorTool())).validate(plan)

    assert not result.ok
    assert "Invalid arguments" in result.errors[0]


def test_cycle_dependency_detection() -> None:
    plan = Plan(
        goal="cycle",
        steps=[
            PlanStep(id="a", description="a", tool_name="echo", arguments={"value": "a"}, depends_on=["b"]),
            PlanStep(id="b", description="b", tool_name="echo", arguments={"value": "b"}, depends_on=["a"]),
        ],
    )

    result = PlanValidator(make_registry(EchoTool([]))).validate(plan)

    assert not result.ok
    assert "cycle" in "; ".join(result.errors)


def test_steps_execute_in_dependency_order(tmp_path: Path) -> None:
    calls: list[str] = []
    registry = make_registry(EchoTool(calls))
    plan = Plan(
        goal="ordered",
        steps=[
            PlanStep(id="first", description="first", tool_name="echo", arguments={"value": "first"}),
            PlanStep(id="second", description="second", tool_name="echo", arguments={"value": "second"}, depends_on=["first"]),
        ],
    )

    result = PlanExecutor(registry=registry, repository=make_repo(tmp_path)).execute(plan)

    assert result.stopped_reason == "completed"
    assert calls == ["first", "second"]


def test_tool_failure_updates_status(tmp_path: Path) -> None:
    registry = make_registry(FailingTool())
    plan = Plan(goal="fail", steps=[PlanStep(id="s1", description="fail", tool_name="fail", arguments={"value": "x"})])

    result = PlanExecutor(registry=registry, repository=make_repo(tmp_path), max_retries=0).execute(plan)

    assert result.stopped_reason == "failed"
    assert result.plan.steps[0].status == StepStatus.FAILED
    assert "boom" in str(result.plan.steps[0].error)


def test_retry_limit_stops_after_retries(tmp_path: Path) -> None:
    registry = make_registry(FailingTool())
    plan = Plan(goal="retry", steps=[PlanStep(id="s1", description="fail", tool_name="fail", arguments={"value": "x"})])

    result = PlanExecutor(registry=registry, repository=make_repo(tmp_path), max_retries=1).execute(plan)

    assert result.stopped_reason == "failed"
    assert result.plan.steps[0].retry_count == 2


def test_replan_limit_stops(tmp_path: Path) -> None:
    registry = make_registry(FailingTool())
    plan = Plan(goal="retry", replan_count=1, steps=[PlanStep(id="s1", description="fail", tool_name="fail", arguments={"value": "x"})])

    result = PlanExecutor(registry=registry, repository=make_repo(tmp_path), max_retries=0, max_replans=1).execute(plan)

    assert result.stopped_reason == "replan_limit_reached"


def test_destructive_operation_requires_confirmation(tmp_path: Path) -> None:
    registry = make_registry(DestructiveTool())
    plan = Plan(goal="danger", steps=[PlanStep(id="s1", description="danger", tool_name="destroy", arguments={"value": "x"})])

    result = PlanExecutor(registry=registry, repository=make_repo(tmp_path)).execute(plan)

    assert result.stopped_reason == "confirmation_required"
    assert result.plan.status == PlanStatus.PAUSED


def test_completed_steps_preserved_on_replan(tmp_path: Path) -> None:
    calls: list[str] = []
    registry = make_registry(EchoTool(calls), FailingTool())
    plan = Plan(
        goal="replan",
        steps=[
            PlanStep(id="done", description="done", tool_name="echo", arguments={"value": "done"}),
            PlanStep(id="bad", description="bad", tool_name="fail", arguments={"value": "bad"}, depends_on=["done"]),
        ],
    )

    def replan_callback(old_plan: Plan, reason: str) -> Plan:
        return Plan(
            goal=old_plan.goal,
            steps=[
                PlanStep(id="done", description="done", tool_name="echo", arguments={"value": "done"}),
                PlanStep(id="fixed", description="fixed", tool_name="echo", arguments={"value": "fixed"}, depends_on=["done"]),
            ],
        )

    result = PlanExecutor(
        registry=registry,
        repository=make_repo(tmp_path),
        max_retries=0,
        replan_callback=replan_callback,
    ).execute(plan)

    assert result.stopped_reason == "completed"
    done = next(step for step in result.plan.steps if step.id == "done")
    assert done.status == StepStatus.COMPLETED
    assert done.actual_output == "done"
    assert calls == ["done", "fixed"]


def test_plan_persists_to_sqlite(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    plan = Plan(goal="persist", steps=[PlanStep(id="s1", description="x", tool_name="echo", arguments={"value": "x"})])

    repo.save(plan)
    loaded = PlanRepository(tmp_path / "plans.sqlite3").get(plan.plan_id)

    assert loaded is not None
    assert loaded.goal == "persist"

