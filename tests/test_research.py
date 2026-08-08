from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import src.planner.executor as executor_module
from src.cli import build_lookup_registry, build_registry, build_research_registry
from src.planner.executor import PlanExecutor
from src.planner.models import Plan, PlanStep, Route, StepStatus
from src.planner.repository import PlanRepository
from src.planner.research import (
    ResearchPlanRaw,
    ResearchSubTopic,
    create_research_plan,
    materialize_research_plan,
)
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.base import ToolExecutionError
from src.tools.cited_report import WriteCitedReportTool
from src.tools.cited_report import WriteCitedReportArguments
from src.tools.registry import ToolRegistry
from src.tools.search_web import SearchWebArguments, SearchWebTool


class StubLLM:
    def __init__(self, response: str = "") -> None:
        self.response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.response


def make_repo(tmp_path: Path) -> PlanRepository:
    return PlanRepository(tmp_path / "plans.sqlite3")


def make_research_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(SearchWebTool("test-key"))
    registry.register(WriteCitedReportTool())
    return registry


def test_search_web_builds_tavily_payload(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"results": [{"title": "X", "url": "https://x.example", "content": "abc" * 200}]}

    def fake_post(url: str, json: dict | None = None, timeout: float | None = None) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("src.tools.search_web.httpx.post", fake_post)

    output = SearchWebTool("test-key").execute(SearchWebArguments(query="hello"))

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["json"]["api_key"] == "test-key"
    assert captured["json"]["query"] == "hello"
    assert captured["json"]["max_results"] == 5
    parsed = json.loads(output)
    assert parsed[0]["url"] == "https://x.example"
    assert len(parsed[0]["content"]) == 500


def test_search_web_requires_api_key() -> None:
    with pytest.raises(ToolExecutionError):
        SearchWebTool().execute(SearchWebArguments(query="hello"))


def test_registry_layers_keep_network_and_report_tools_separate() -> None:
    core_tools = {tool["name"] for tool in build_registry().list_tools()}
    lookup_tools = {tool["name"] for tool in build_lookup_registry("test-key").list_tools()}
    research_tools = {tool["name"] for tool in build_research_registry("test-key").list_tools()}

    assert "search_web" not in core_tools
    assert "write_cited_report" not in core_tools
    assert "search_web" in lookup_tools
    assert "write_cited_report" not in lookup_tools
    assert "search_web" in research_tools
    assert "write_cited_report" in research_tools


def test_search_web_raises_on_http_error(monkeypatch) -> None:
    def fake_post(url: str, json: dict | None = None, timeout: float | None = None) -> None:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr("src.tools.search_web.httpx.post", fake_post)

    with pytest.raises(ToolExecutionError):
        SearchWebTool("key").execute(SearchWebArguments(query="hello"))


def test_research_request_routes_deep_research() -> None:
    decision = RequestRouter().route("帮我调研一下 2025 年 AI Agent 框架的对比")

    assert decision.route == Route.DEEP_RESEARCH


def test_plain_research_word_not_hijacked() -> None:
    decision = RequestRouter().route("我想研究一下这个报错")

    assert decision.route != Route.DEEP_RESEARCH


def test_materialize_research_plan_builds_steps() -> None:
    raw = ResearchPlanRaw(
        topic="AI Agent 框架",
        sub_topics=[
            ResearchSubTopic(question="有哪些主流框架", search_queries=["AI agent 框架 对比"]),
            ResearchSubTopic(question="社区活跃度", search_queries=["AI agent framework github"]),
        ],
        report_file="reports/agents.md",
    )

    plan = materialize_research_plan(raw, max_steps=8)

    assert [step.tool_name for step in plan.steps] == ["search_web", "search_web", "write_cited_report"]
    assert plan.steps[-1].depends_on == ["step_1", "step_2"]
    assert plan.steps[-1].arguments["report_file"] == "reports/agents.md"
    assert plan.unresolved_questions == []


def test_materialize_research_plan_clamps_total_queries() -> None:
    raw = ResearchPlanRaw(
        topic="t",
        sub_topics=[ResearchSubTopic(question=f"q{i}", search_queries=["a", "b"]) for i in range(5)],
    )

    plan = materialize_research_plan(raw, max_steps=4)

    assert len(plan.steps) == 4
    assert sum(1 for step in plan.steps if step.tool_name == "search_web") == 3
    assert plan.steps[-1].tool_name == "write_cited_report"


def test_materialize_research_plan_requires_subtopics() -> None:
    with pytest.raises(ValueError):
        materialize_research_plan(ResearchPlanRaw(topic="t", sub_topics=[]), max_steps=8)


def test_create_research_plan_uses_llm() -> None:
    llm = StubLLM(
        json.dumps(
            {
                "topic": "AI Agent 框架",
                "sub_topics": [{"question": "主流框架", "search_queries": ["AI agent 框架"]}],
                "report_file": "reports/agents.md",
                "assumptions": [],
            },
            ensure_ascii=False,
        )
    )

    plan = create_research_plan(llm, make_research_registry(), "调研一下", max_steps=8)

    assert plan.steps[0].tool_name == "search_web"
    assert plan.steps[-1].tool_name == "write_cited_report"
    assert plan.steps[-1].depends_on == ["step_1"]


def test_executor_writes_cited_report(tmp_path: Path, monkeypatch) -> None:
    search_output = json.dumps(
        [{"title": "T", "url": "https://example.com", "content": "c"}], ensure_ascii=False
    )
    plan = Plan(
        goal="深度调研：test",
        steps=[
            PlanStep(
                id="step_1",
                description="搜索：sub",
                tool_name="search_web",
                arguments={"query": "q", "max_results": 5},
                status=StepStatus.COMPLETED,
                actual_output=search_output,
            ),
            PlanStep(
                id="step_2",
                description="报告",
                tool_name="write_cited_report",
                arguments={"topic": "test", "report_file": "reports/test.md", "sub_topics": ["sub"]},
                depends_on=["step_1"],
            ),
        ],
    )
    monkeypatch.setattr(executor_module, "resolve_workspace_path", lambda p: tmp_path / p)
    llm = StubLLM("# 报告\n正文 [来源](https://example.com)")

    result = PlanExecutor(
        registry=make_research_registry(),
        repository=make_repo(tmp_path),
        llm_client=llm,
        max_retries=0,
        confirmation_callback=lambda plan, step, risk, reason: True,
    ).execute(plan)

    assert result.stopped_reason == "completed"
    report = tmp_path / "reports" / "test.md"
    assert report.exists()
    assert "# 报告" in report.read_text(encoding="utf-8")
    assert "报告已保存到" in str(result.plan.steps[1].actual_output)


def test_research_plan_validates_against_research_registry() -> None:
    plan = Plan(
        goal="深度调研：test",
        steps=[
            PlanStep(
                id="step_1",
                description="搜索",
                tool_name="search_web",
                arguments={"query": "q", "max_results": 5},
            ),
            PlanStep(
                id="step_2",
                description="报告",
                tool_name="write_cited_report",
                arguments={"topic": "t", "report_file": "reports/t.md", "sub_topics": []},
                depends_on=["step_1"],
            ),
        ],
    )

    validation = PlanValidator(make_research_registry()).validate(plan)

    assert validation.ok


def test_cited_report_rejects_paths_outside_reports() -> None:
    with pytest.raises(ValueError):
        WriteCitedReportArguments(topic="x", report_file="../x.md")
    with pytest.raises(ValueError):
        WriteCitedReportArguments(topic="x", report_file="x.md")
