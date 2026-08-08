from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from src.agent.protocol import extract_first_json_object
from src.planner.models import Route, RouteDecision
from src.planner.prompts import build_router_prompt


class RouterLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class RouteCandidate:
    decision: RouteDecision
    confidence: float
    final: bool = False
    source: str = ""


class ConstraintRouter:
    def route(self, user_input: str) -> RouteCandidate | None:
        text = user_input.strip().lower()
        if self._missing_information(text):
            return RouteCandidate(
                RouteDecision(
                    route=Route.CLARIFICATION,
                    reason="约束路由：缺少执行所必需的文件名、目标或写入位置。",
                    missing_information=["请提供缺失的关键参数，例如文件名、输入内容或保存位置。"],
                ),
                confidence=1.0,
                final=True,
                source="constraint",
            )
        if self._requires_current_information(text):
            return RouteCandidate(
                RouteDecision(route=Route.WEB_LOOKUP, reason="启发式路由：请求依赖最新或当前外部信息，但未要求系统调研。"),
                confidence=0.72,
                final=False,
                source="constraint",
            )
        if self._looks_research(text):
            return RouteCandidate(
                RouteDecision(route=Route.DEEP_RESEARCH, reason="约束路由：请求明确要求联网调研或研究报告。"),
                confidence=0.95,
                final=True,
                source="constraint",
            )
        if self._looks_planned(text):
            return RouteCandidate(
                RouteDecision(route=Route.PLANNED_TASK, reason="约束路由：请求包含多个依赖步骤。"),
                confidence=0.9,
                final=True,
                source="constraint",
            )
        if self._looks_single_tool(text):
            return RouteCandidate(
                RouteDecision(route=Route.SINGLE_TOOL, reason="约束路由：请求明显只需要一次确定性工具调用。"),
                confidence=0.9,
                final=True,
                source="constraint",
            )
        return None

    def fallback(self, user_input: str) -> RouteDecision:
        candidate = self.route(user_input)
        if candidate:
            return candidate.decision
        return RouteDecision(route=Route.DIRECT_ANSWER, reason="规则兜底：普通对话或可直接回答。")

    def _missing_information(self, text: str) -> bool:
        vague_actions = ("保存一下", "写个文件", "读取文件", "帮我处理这个")
        return any(action in text for action in vague_actions) and not re.search(r"[\w\-]+\.txt|[\w\-]+\.md", text)

    def _requires_current_information(self, text: str) -> bool:
        simple_external_patterns = (
            r"\blatest\b.*\b(price|pricing|status|version|release|news|api)\b",
            r"\bcurrent\b.*\b(ceo|cto|president|prime minister|price|pricing|status|version|release|news)\b",
            r"\bwho is\b.*\bcurrent\b",
            r"\bwhat is\b.*\blatest\b",
            r"(当前|现在).*(ceo|cto|负责人|总统|主席|价格|状态|版本|新闻)",
            r"(最新|最近).*(价格|状态|版本|发布|新闻|api)",
        )
        return any(re.search(pattern, text) for pattern in simple_external_patterns)

    def _looks_research(self, text: str) -> bool:
        chinese_markers = ("调研", "深度研究", "研究报告", "调查报告")
        english_markers = ("deep research", "research report", "investigate online", "web research", "deep dive")
        comparison_markers = ("比较", "对比", "优缺点", "pros and cons", "compare")
        research_intent = any(marker in text for marker in chinese_markers) or any(marker in text for marker in english_markers)
        return research_intent or (("research" in text or "调研" in text) and any(marker in text for marker in comparison_markers))

    def _looks_single_tool(self, text: str) -> bool:
        has_file_path = bool(re.search(r"[\w\-]+\.(txt|md)", text))
        calculator = bool(
            re.search(r"\bcalculate\b", text)
            or "计算" in text
            or re.search(r"\d+(?:\.\d+)?\s*(?:\+|\*|/|×|÷)\s*\d+(?:\.\d+)?", text)
            or re.search(r"\d+(?:\.\d+)?\s+-\s+\d+(?:\.\d+)?", text)
        )
        read = ("读取" in text or "read" in text) and has_file_path and ("保存" not in text and "写入" not in text)
        write = ("写入" in text or "保存" in text or "write" in text) and has_file_path and not ("读取" in text or "read" in text)
        return calculator or read or write

    def _looks_planned(self, text: str) -> bool:
        markers = ("然后", "并", "再", "整理", "总结", "保存", "写入", "生成", "找出")
        return sum(1 for marker in markers if marker in text) >= 2 or (
            ("读取" in text or "read" in text) and ("保存" in text or "写入" in text or "write" in text)
        )


class SemanticRouter:
    def __init__(self, *, threshold: float = 0.42) -> None:
        self.threshold = threshold
        self.examples: dict[Route, tuple[str, ...]] = {
            Route.WEB_LOOKUP: (
                "latest api pricing simple lookup",
                "current ceo who is simple external fact",
                "最新 价格 当前 ceo 简单 查询",
            ),
            Route.DEEP_RESEARCH: (
                "compare recent tools with multiple web sources",
                "systematic research report pros and cons",
                "调研 比较 优缺点 研究报告 多来源",
            ),
            Route.PLANNED_TASK: (
                "read file summarize and save result",
                "multi step transform then write file",
                "读取 文件 整理 保存 多步骤",
            ),
            Route.SINGLE_TOOL: (
                "calculate arithmetic expression",
                "read one workspace file",
                "write one workspace file",
                "计算 读取 单个 文件",
            ),
        }

    def route(self, user_input: str) -> RouteCandidate | None:
        query_tokens = self._tokens(user_input)
        if not query_tokens:
            return None
        best_route: Route | None = None
        best_score = 0.0
        for route, examples in self.examples.items():
            for example in examples:
                score = self._jaccard(query_tokens, self._tokens(example))
                if score > best_score:
                    best_score = score
                    best_route = route
        if best_route is None or best_score < self.threshold:
            return None
        return RouteCandidate(
            RouteDecision(route=best_route, reason=f"语义路由：与 {best_route} 示例相似度 {best_score:.2f}。"),
            confidence=best_score,
            final=False,
            source="semantic",
        )

    def _tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2}", lowered))
        chinese_bigrams = {
            lowered[index : index + 2]
            for index in range(max(0, len(lowered) - 1))
            if re.fullmatch(r"[\u4e00-\u9fff]{2}", lowered[index : index + 2])
        }
        return tokens | chinese_bigrams

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)


class LLMRouter:
    def __init__(self, llm_client: RouterLLM) -> None:
        self.llm_client = llm_client

    def _route_with_llm(self, user_input: str, *, memory_context: str) -> RouteDecision | None:
        raw = self.llm_client.chat(
            [{"role": "user", "content": build_router_prompt(user_input, memory_context)}]
        )
        json_text = extract_first_json_object(raw)
        if json_text is None:
            return None
        try:
            return RouteDecision.model_validate_json(json_text)
        except ValidationError:
            return None


class RequestRouter:
    def __init__(
        self,
        llm_client: RouterLLM | None = None,
        *,
        constraint_router: ConstraintRouter | None = None,
        semantic_router: SemanticRouter | None = None,
    ) -> None:
        self.constraint_router = constraint_router or ConstraintRouter()
        self.semantic_router = semantic_router or SemanticRouter()
        self.llm_router = LLMRouter(llm_client) if llm_client is not None else None

    def route(self, user_input: str, *, memory_context: str = "") -> RouteDecision:
        constraint = self.constraint_router.route(user_input)
        if constraint and constraint.final:
            return constraint.decision

        semantic = self.semantic_router.route(user_input)
        llm_decision = None
        if self.llm_router is not None:
            llm_decision = self.llm_router._route_with_llm(user_input, memory_context=memory_context)
            if llm_decision is not None:
                if (
                    constraint is not None
                    and constraint.decision.route == Route.WEB_LOOKUP
                    and llm_decision.route == Route.DIRECT_ANSWER
                ):
                    return constraint.decision
                return llm_decision

        if constraint is not None:
            return constraint.decision
        if semantic is not None:
            return semantic.decision
        return self.constraint_router.fallback(user_input)
