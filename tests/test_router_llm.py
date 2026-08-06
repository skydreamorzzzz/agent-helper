from __future__ import annotations

from src.planner.models import Route
from src.planner.router import RequestRouter


class StubLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.response


def test_llm_routes_factual_question_to_deep_research() -> None:
    llm = StubLLM('{"route":"deep_research","reason":"needs web","missing_information":[]}')

    decision = RequestRouter(llm).route("The total number of countries in 2025?")

    assert decision.route == Route.DEEP_RESEARCH


def test_llm_parse_failure_falls_back_to_rules() -> None:
    decision = RequestRouter(StubLLM("not json at all")).route("计算 23 * 7")

    assert decision.route == Route.SINGLE_TOOL


def test_explicit_research_keyword_wins_over_llm() -> None:
    llm = StubLLM('{"route":"direct_answer","reason":"miscategorized","missing_information":[]}')

    decision = RequestRouter(llm).route("帮我调研一下 AI Agent 框架")

    assert decision.route == Route.DEEP_RESEARCH


def test_single_tool_not_fooled_by_date_like_strings() -> None:
    decision = RequestRouter().route("What happened in 2023-01-01?")

    assert decision.route != Route.SINGLE_TOOL


def test_single_tool_not_fooled_by_bare_read_in_english() -> None:
    decision = RequestRouter().route("Please read the situation and tell me.")

    assert decision.route != Route.SINGLE_TOOL
