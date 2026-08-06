from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, Field

from src.planner.models import RiskLevel
from src.tools.base import Tool, ToolExecutionError

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
RESULT_CONTENT_CHARS = 500


class SearchWebArguments(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class SearchWebTool(Tool):
    name = "search_web"
    description = (
        "Search the web with Tavily and return a JSON list of results with title, url, and a short content snippet. "
        "Use for fact-finding or gathering sources on a topic."
    )
    argument_schema = SearchWebArguments
    risk_level = RiskLevel.READ_ONLY

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    def execute(self, arguments: SearchWebArguments) -> str:
        if not self.api_key:
            raise ToolExecutionError("TAVILY_API_KEY 未配置，无法联网搜索。")
        try:
            response = httpx.post(
                TAVILY_SEARCH_URL,
                json={
                    "api_key": self.api_key,
                    "query": arguments.query,
                    "max_results": arguments.max_results,
                    "search_depth": "basic",
                },
                timeout=15.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"Tavily 搜索请求失败：{type(exc).__name__}: {exc}") from exc

        payload = response.json()
        results = payload.get("results") or []
        if not results:
            raise ToolExecutionError(f"搜索「{arguments.query}」没有返回任何结果。")

        compact = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:RESULT_CONTENT_CHARS],
            }
            for item in results
        ]
        return json.dumps(compact, ensure_ascii=False)
