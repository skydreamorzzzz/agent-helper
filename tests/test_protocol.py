from __future__ import annotations

import pytest

from src.agent.protocol import FinalAnswer, ModelOutputParseError, ToolCall, parse_model_output


def test_parse_model_output_tool_call_json() -> None:
    parsed = parse_model_output(
        '{"type":"tool_call","tool":"calculator","arguments":{"expression":"1+2"}}'
    )

    assert isinstance(parsed, ToolCall)
    assert parsed.tool == "calculator"
    assert parsed.arguments == {"expression": "1+2"}


def test_parse_model_output_markdown_json() -> None:
    parsed = parse_model_output(
        '```json\n{"type":"final_answer","content":"done"}\n```'
    )

    assert isinstance(parsed, FinalAnswer)
    assert parsed.content == "done"


def test_parse_model_output_embedded_json() -> None:
    parsed = parse_model_output(
        '<think>planning</think>\n{"type":"tool_call","tool":"calculator","arguments":{"expression":"2*3"}}'
    )

    assert isinstance(parsed, ToolCall)
    assert parsed.arguments == {"expression": "2*3"}


def test_parse_model_output_rejects_invalid_json() -> None:
    with pytest.raises(ModelOutputParseError):
        parse_model_output("not json")
