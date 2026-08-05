from __future__ import annotations

import pytest

from src.tools.calculator import CalculatorArguments, CalculatorTool
from src.tools.file_tools import resolve_workspace_path
from src.tools.registry import ToolRegistry, UnknownToolError
from src.tools.transform_text import TransformTextTool


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
