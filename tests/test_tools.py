from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.planner.models import RiskLevel
from src.tools.base import Tool
from src.tools.calculator import CalculatorArguments, CalculatorTool
from src.tools.file_tools import resolve_workspace_path
from src.tools.policy import PolicyAction, ToolExecutionPolicy
from src.tools.registry import ToolRegistry, UnknownToolError
from src.tools.transform_text import TransformTextTool


class DummyArguments(BaseModel):
    path: str = "x"
    overwrite: bool = False


class DummyWriteTool(Tool):
    name = "write_text_file"
    description = "dummy write"
    argument_schema = DummyArguments
    risk_level = RiskLevel.WRITE

    def execute(self, arguments: DummyArguments) -> str:
        return "ok"


class DummyDestructiveTool(DummyWriteTool):
    name = "destroy"
    risk_level = RiskLevel.DESTRUCTIVE


def test_calculator_evaluates_basic_math() -> None:
    tool = CalculatorTool()

    result = tool.execute(CalculatorArguments(expression="23.5 * 17"))

    assert result == "399.5"


def test_calculator_rejects_malicious_expression() -> None:
    tool = CalculatorTool()

    with pytest.raises(ValueError):
        tool.execute(CalculatorArguments(expression="__import__('os').system('whoami')"))


def test_file_tools_reject_path_traversal() -> None:
    with pytest.raises(ValueError):
        resolve_workspace_path("../outside.txt")


def test_tool_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError):
        registry.get("missing_tool")


def test_transform_text_declares_read_only_risk() -> None:
    assert TransformTextTool().risk_level == "read_only"


def test_tool_execution_policy_write_confirm_is_shared_decision() -> None:
    decision = ToolExecutionPolicy().evaluate(
        tool=DummyWriteTool(),
        arguments={"path": "a.txt", "overwrite": False},
        confirm_write_actions=True,
    )

    assert decision.action == PolicyAction.CONFIRM
    assert decision.risk_level == RiskLevel.WRITE


def test_tool_execution_policy_overwrite_and_destructive_require_confirmation() -> None:
    policy = ToolExecutionPolicy()

    overwrite = policy.evaluate(
        tool=DummyWriteTool(),
        arguments={"path": "a.txt", "overwrite": True},
        confirm_write_actions=False,
    )
    destructive = policy.evaluate(
        tool=DummyDestructiveTool(),
        arguments={"path": "x"},
        confirm_write_actions=False,
    )

    assert overwrite.action == PolicyAction.CONFIRM
    assert "overwrite" in overwrite.reason
    assert destructive.action == PolicyAction.CONFIRM
    assert "destructive" in destructive.reason
