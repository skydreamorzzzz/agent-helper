from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field

from src.agent.runtime import AgentRuntime
from src.planner.models import RiskLevel
from src.tools.calculator import CalculatorTool
from src.tools.base import Tool
from src.tools.registry import ToolRegistry


class MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(copy.deepcopy(messages))
        if not self.responses:
            raise AssertionError("No mock response left")
        return self.responses.pop(0)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)


class WriteArgs(BaseModel):
    path: str
    content: str
    overwrite: bool = False


class FakeWriteTextFileTool(Tool):
    name = "write_text_file"
    description = "Fake write tool for runtime tests."
    argument_schema = WriteArgs
    risk_level = RiskLevel.WRITE

    def __init__(self) -> None:
        self.calls: list[WriteArgs] = []

    def execute(self, arguments: WriteArgs) -> str:
        self.calls.append(arguments)
        return f"wrote {arguments.path}"


class FakeDestructiveTool(FakeWriteTextFileTool):
    name = "destroy"
    risk_level = RiskLevel.DESTRUCTIVE


class FakeSearchTool(Tool):
    name = "search_web"
    description = "Fake search tool for runtime tests."
    argument_schema = SearchArgs

    def __init__(self) -> None:
        self.calls: list[SearchArgs] = []

    def execute(self, arguments: SearchArgs) -> Any:
        self.calls.append(arguments)
        return '[{"title":"Tavily pricing","url":"https://example.com","content":"pricing info"}]'


class FailingSearchTool(FakeSearchTool):
    def execute(self, arguments: SearchArgs) -> Any:
        self.calls.append(arguments)
        raise RuntimeError("search failed")


def build_lookup_registry(search_tool: FakeSearchTool) -> ToolRegistry:
    registry = build_registry()
    registry.register(search_tool)
    return registry


def test_read_only_calculator_executes_without_confirmation() -> None:
    runtime = AgentRuntime(
        llm_client=MockLLM(
            [
                '{"type":"tool_call","tool":"calculator","arguments":{"expression":"23 * 7"}}',
                '{"type":"final_answer","content":"161"}',
            ]
        ),
        tool_registry=build_registry(),
        confirmation_callback=lambda tool, args, risk, reason: False,
    )

    result = runtime.run("计算 23 * 7")

    assert result.stopped_reason == "final_answer"
    assert result.content == "161"


def test_write_text_file_runtime_requires_confirmation_before_execution() -> None:
    write_tool = FakeWriteTextFileTool()
    registry = build_registry()
    registry.register(write_tool)
    confirmations: list[tuple[str, RiskLevel, str]] = []

    runtime = AgentRuntime(
        llm_client=MockLLM(
            [
                '{"type":"tool_call","tool":"write_text_file","arguments":{"path":"a.txt","content":"hello"}}',
                '{"type":"final_answer","content":"written"}',
            ]
        ),
        tool_registry=registry,
        confirmation_callback=lambda tool, args, risk, reason: confirmations.append((tool, risk, reason)) or True,
    )

    result = runtime.run("写入 a.txt")

    assert result.stopped_reason == "final_answer"
    assert [call.path for call in write_tool.calls] == ["a.txt"]
    assert confirmations == [("write_text_file", RiskLevel.WRITE, "write action requires confirmation")]


def test_write_text_file_runtime_rejection_prevents_execution() -> None:
    write_tool = FakeWriteTextFileTool()
    registry = build_registry()
    registry.register(write_tool)

    runtime = AgentRuntime(
        llm_client=MockLLM(
            ['{"type":"tool_call","tool":"write_text_file","arguments":{"path":"a.txt","content":"hello"}}']
        ),
        tool_registry=registry,
        confirmation_callback=lambda tool, args, risk, reason: False,
    )

    result = runtime.run("写入 a.txt")

    assert result.stopped_reason == "tool_confirmation_rejected"
    assert write_tool.calls == []
    assert "requires confirmation" in result.content


def test_destructive_runtime_tool_requires_confirmation() -> None:
    destructive_tool = FakeDestructiveTool()
    registry = build_registry()
    registry.register(destructive_tool)

    runtime = AgentRuntime(
        llm_client=MockLLM(['{"type":"tool_call","tool":"destroy","arguments":{"path":"x","content":"boom"}}']),
        tool_registry=registry,
    )

    result = runtime.run("destroy")

    assert result.stopped_reason == "tool_confirmation_rejected"
    assert destructive_tool.calls == []
    assert "destructive action requires confirmation" in result.content


def test_web_lookup_required_tool_rejects_final_until_search_succeeds() -> None:
    search_tool = FakeSearchTool()
    llm = MockLLM(
        [
            '{"type":"final_answer","content":"Skipped search."}',
            '{"type":"tool_call","tool":"search_web","arguments":{"query":"latest Tavily API pricing"}}',
            '{"type":"final_answer","content":"Tavily pricing was checked via search."}',
        ]
    )
    runtime = AgentRuntime(
        llm_client=llm,
        tool_registry=build_lookup_registry(search_tool),
    )

    result = runtime.run(
        "What is the latest Tavily API pricing?",
        required_tool="search_web",
        execution_policy=(
            "This request was routed as web_lookup.\n"
            "You must call search_web before returning a final answer."
        ),
    )

    assert result.stopped_reason == "final_answer"
    assert result.content == "Tavily pricing was checked via search."
    assert [call.query for call in search_tool.calls] == ["latest Tavily API pricing"]
    assert "requires a successful search_web tool call" in llm.calls[1][-1]["content"]
    assert "Tool result JSON" in llm.calls[2][-1]["content"]


def test_required_tool_failure_does_not_allow_final_answer() -> None:
    search_tool = FailingSearchTool()
    runtime = AgentRuntime(
        llm_client=MockLLM(
            [
                '{"type":"tool_call","tool":"search_web","arguments":{"query":"latest Tavily API pricing"}}',
                '{"type":"final_answer","content":"I searched."}',
                '{"type":"final_answer","content":"Still final."}',
            ]
        ),
        tool_registry=build_lookup_registry(search_tool),
    )

    result = runtime.run(
        "What is the latest Tavily API pricing?",
        required_tool="search_web",
        execution_policy="This request was routed as web_lookup. You must call search_web before returning a final answer.",
    )

    assert result.stopped_reason == "required_tool_missing"
    assert [call.query for call in search_tool.calls] == ["latest Tavily API pricing"]
    assert "必须先成功调用 search_web" in result.content


def test_core_runtime_still_accepts_first_turn_final_answer() -> None:
    runtime = AgentRuntime(
        llm_client=MockLLM(['{"type":"final_answer","content":"hello"}']),
        tool_registry=build_registry(),
    )

    result = runtime.run("hello")

    assert result.stopped_reason == "final_answer"
    assert result.content == "hello"


def test_invalid_json_repair_failure_stops_safely() -> None:
    runtime = AgentRuntime(
        llm_client=MockLLM(["not json", "still not json"]),
        tool_registry=build_registry(),
    )

    result = runtime.run("hello")

    assert result.stopped_reason == "invalid_json_repair_failed"
    assert "格式修复失败" in result.content


def test_agent_stops_at_max_tool_calls() -> None:
    runtime = AgentRuntime(
        llm_client=MockLLM(
            [
                '{"type":"tool_call","tool":"calculator","arguments":{"expression":"1+1"}}',
                '{"type":"tool_call","tool":"calculator","arguments":{"expression":"2+2"}}',
            ]
        ),
        tool_registry=build_registry(),
        max_tool_calls=1,
    )

    result = runtime.run("keep calculating")

    assert result.stopped_reason == "max_tool_calls_reached"
    assert "最大工具调用轮数 1" in result.content


def test_llm_exception_stops_safely() -> None:
    class FailingLLM:
        def chat(self, messages: list[dict[str, str]]) -> str:
            raise TimeoutError("slow model")

    runtime = AgentRuntime(
        llm_client=FailingLLM(),
        tool_registry=build_registry(),
    )

    result = runtime.run("hello")

    assert result.stopped_reason == "llm_call_failed"
    assert "模型调用失败" in result.content
