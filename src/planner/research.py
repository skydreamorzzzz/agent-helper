from __future__ import annotations

import re
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from src.agent.protocol import ModelOutputParseError, extract_first_json_object
from src.planner.models import Plan, PlanStep
from src.planner.prompts import build_research_plan_prompt
from src.tools.registry import ToolRegistry


class ResearchLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class ResearchSubTopic(BaseModel):
    question: str
    search_queries: list[str] = Field(default_factory=list)


class ResearchPlanRaw(BaseModel):
    topic: str
    sub_topics: list[ResearchSubTopic]
    report_file: str = ""
    assumptions: list[str] = Field(default_factory=list)


def create_research_plan(
    llm_client: ResearchLLM,
    registry: ToolRegistry,
    user_input: str,
    *,
    memory_context: str = "",
    max_steps: int = 8,
) -> Plan:
    raw = llm_client.chat(
        [
            {
                "role": "user",
                "content": build_research_plan_prompt(
                    user_input=user_input,
                    registry=registry,
                    memory_context=memory_context,
                    max_steps=max_steps,
                ),
            }
        ]
    )
    json_text = extract_first_json_object(raw)
    if json_text is None:
        raise ModelOutputParseError("Research planner returned no JSON object")
    try:
        raw_plan = ResearchPlanRaw.model_validate_json(json_text)
    except ValidationError as exc:
        raise ModelOutputParseError(str(exc)) from exc
    return materialize_research_plan(raw_plan, max_steps=max_steps)


def materialize_research_plan(raw: ResearchPlanRaw, *, max_steps: int = 8) -> Plan:
    if not raw.sub_topics:
        raise ValueError("研究计划未生成任何子主题，请换一种表述重试。")

    max_search_steps = max(1, max_steps - 1)
    steps: list[PlanStep] = []
    search_ids: list[str] = []
    kept_subtopics: list[str] = []

    for sub in raw.sub_topics:
        if len(steps) >= max_search_steps:
            break
        kept_questions: list[str] = []
        for query in sub.search_queries:
            if not query.strip():
                continue
            if len(steps) >= max_search_steps:
                break
            steps.append(
                PlanStep(
                    id=f"step_{len(steps) + 1}",
                    description=f"搜索：{sub.question}",
                    tool_name="search_web",
                    arguments={"query": query.strip(), "max_results": 5},
                    depends_on=[],
                    expected_output="搜索结果（标题/链接/摘要）",
                )
            )
            search_ids.append(steps[-1].id)
            kept_questions.append(sub.question)
        if kept_questions:
            kept_subtopics.append(sub.question)

    if not steps:
        raise ValueError("研究计划未生成任何搜索步骤。")

    steps.append(
        PlanStep(
            id=f"step_{len(steps) + 1}",
            description="综合所有搜索材料，生成带引用的中文调研报告并保存",
            tool_name="write_cited_report",
            arguments={
                "topic": raw.topic,
                "report_file": raw.report_file or _default_report_filename(raw.topic),
                "sub_topics": kept_subtopics,
            },
            depends_on=list(search_ids),
            expected_output="已保存到 workspace/reports/ 的 Markdown 报告",
        )
    )

    return Plan(
        goal=f"深度调研：{raw.topic}",
        steps=steps,
        assumptions=raw.assumptions,
        final_output_requirement=(
            "输出一份带引用来源的中文 Markdown 调研报告，保存到 workspace/reports/ 下，并给出简短总结。"
        ),
    )


def _default_report_filename(topic: str) -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    slug = re.sub(r"[^\w一-鿿]+", "_", topic).strip("_")[:40]
    return f"reports/research_{slug or 'topic'}_{date_part}.md"
