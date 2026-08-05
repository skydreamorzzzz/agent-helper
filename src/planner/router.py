from __future__ import annotations

import re
from typing import Protocol

from pydantic import ValidationError

from src.agent.protocol import extract_first_json_object
from src.planner.models import Route, RouteDecision
from src.planner.prompts import build_router_prompt


class RouterLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class RequestRouter:
    def __init__(self, llm_client: RouterLLM | None = None) -> None:
        self.llm_client = llm_client

    def route(self, user_input: str, *, memory_context: str = "") -> RouteDecision:
        if self.llm_client is not None:
            decision = self._route_with_llm(user_input, memory_context=memory_context)
            if decision is not None:
                return decision
        return self._route_with_rules(user_input)

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

    def _route_with_rules(self, user_input: str) -> RouteDecision:
        text = user_input.strip().lower()
        if self._missing_information(text):
            return RouteDecision(
                route=Route.CLARIFICATION,
                reason="缺少执行所必需的文件名、目标或写入位置。",
                missing_information=["请提供缺失的关键参数，例如文件名、输入内容或保存位置。"],
            )
        if self._looks_planned(text):
            return RouteDecision(route=Route.PLANNED_TASK, reason="请求包含多个依赖步骤。")
        if self._looks_single_tool(text):
            return RouteDecision(route=Route.SINGLE_TOOL, reason="请求明显只需要一次工具调用。")
        return RouteDecision(route=Route.DIRECT_ANSWER, reason="普通对话或可直接回答。")

    def _looks_single_tool(self, text: str) -> bool:
        calculator = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+|计算|calculate", text))
        read = ("读取" in text or "read" in text) and ("保存" not in text and "写入" not in text)
        write = ("写入" in text or "保存" in text or "write" in text) and not ("读取" in text or "read" in text)
        return calculator or read or write

    def _looks_planned(self, text: str) -> bool:
        markers = ("然后", "并", "再", "整理", "总结", "保存", "写入", "生成", "找出")
        return sum(1 for marker in markers if marker in text) >= 2 or (
            ("读取" in text or "read" in text) and ("保存" in text or "写入" in text or "write" in text)
        )

    def _missing_information(self, text: str) -> bool:
        vague_actions = ("保存一下", "写个文件", "读取文件", "帮我处理这个")
        return any(action in text for action in vague_actions) and not re.search(r"[\w\-]+\.txt|[\w\-]+\.md", text)

