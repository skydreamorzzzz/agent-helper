from __future__ import annotations

from src.planner.models import Route
from src.planner.router import RequestRouter, SemanticRouter


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


def test_constraint_single_tool_wins_over_bad_llm_route() -> None:
    llm = StubLLM('{"route":"deep_research","reason":"bad call","missing_information":[]}')

    decision = RequestRouter(llm).route("计算 23 * 7")

    assert decision.route == Route.SINGLE_TOOL


def test_bare_url_is_not_calculator_single_tool() -> None:
    decision = RequestRouter().route("请解释 https://example.com/path 这个链接大概是什么")

    assert decision.route != Route.SINGLE_TOOL


def test_english_research_word_alone_does_not_force_deep_research() -> None:
    decision = RequestRouter().route("I am writing a research methods section for my paper.")

    assert decision.route != Route.DEEP_RESEARCH


def test_current_information_constraint_routes_to_web_lookup_without_llm() -> None:
    decision = RequestRouter().route("What is the latest Tavily API pricing?")

    assert decision.route == Route.WEB_LOOKUP


def test_current_ceo_routes_to_web_lookup() -> None:
    decision = RequestRouter().route("Who is the current CEO of OpenAI?")

    assert decision.route == Route.WEB_LOOKUP


def test_web_lookup_soft_constraint_beats_bad_direct_llm_route() -> None:
    llm = StubLLM('{"route":"direct_answer","reason":"bad stale answer","missing_information":[]}')

    decision = RequestRouter(llm).route("What is the latest Tavily API pricing?")

    assert decision.route == Route.WEB_LOOKUP


def test_software_version_explanation_stays_direct_answer() -> None:
    decision = RequestRouter().route("解释一下软件版本号是什么")

    assert decision.route == Route.DIRECT_ANSWER


def test_memory_methods_comparison_routes_to_deep_research() -> None:
    decision = RequestRouter().route("调研主流 Agent Memory 方法并比较")

    assert decision.route == Route.DEEP_RESEARCH


def test_research_intent_wins_over_current_information_soft_signal() -> None:
    decision = RequestRouter().route("帮我调研最新 Tavily API 价格并比较不同套餐")

    assert decision.route == Route.DEEP_RESEARCH


def test_semantic_router_can_route_similar_planned_task() -> None:
    candidate = SemanticRouter(threshold=0.2).route("multi step transform then write file")

    assert candidate is not None
    assert candidate.decision.route == Route.PLANNED_TASK
