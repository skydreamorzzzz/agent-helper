from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel

from src.planner.models import RiskLevel


class ToolExecutionError(RuntimeError):
    pass


class Tool(ABC):
    name: str
    description: str
    argument_schema: Type[BaseModel]
    risk_level: RiskLevel = RiskLevel.READ_ONLY

    @abstractmethod
    def execute(self, arguments: BaseModel) -> Any:
        raise NotImplementedError

    def schema_for_prompt(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "arguments_json_schema": self.argument_schema.model_json_schema(),
        }
