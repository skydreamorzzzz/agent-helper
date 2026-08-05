from __future__ import annotations

from src.agent.runtime import AgentRuntime
from src.tools.calculator import CalculatorTool
from src.tools.registry import ToolRegistry


class MockLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("No mock response left")
        return self.responses.pop(0)


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


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
