from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field

from src.agent.runtime import AgentRuntime
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


class FakeSearchTool(Tool):
    name = "search_web"
    description = "Fake search tool for runtime tests."
    argument_schema = SearchArgs

    def __init__(self) -> None:
        self.calls: list[SearchArgs] = []

    def execute(self, arguments: SearchArgs) -> Any:
        self.calls.append(arguments)
        return '[{"title":"Tavily pricing","url":"https://example.com","content":"pricing info"}]'


def build_lookup_registry(search_tool: FakeSearchTool) -> ToolRegistry:
    registry = build_registry()
    registry.register(search_tool)
    return registry


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
