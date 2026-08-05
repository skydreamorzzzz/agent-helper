from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from src.tools.base import Tool


class UnknownToolError(ValueError):
    pass


class ToolArgumentError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def list_tools(self) -> list[dict[str, Any]]:
        return [tool.schema_for_prompt() for tool in self._tools.values()]

    def describe_tools(self) -> str:
        return json.dumps(self.list_tools(), ensure_ascii=False, indent=2)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        try:
            validated = tool.argument_schema.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentError(str(exc)) from exc

        try:
            result = tool.execute(validated)
            return {"ok": True, "result": result}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

