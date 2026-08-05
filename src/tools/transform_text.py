from __future__ import annotations

from pydantic import BaseModel, Field

from src.planner.models import RiskLevel
from src.tools.base import Tool


class TransformTextArguments(BaseModel):
    input_text: str = Field(min_length=1, max_length=20000)
    instruction: str = Field(min_length=1, max_length=2000)


class TransformTextTool(Tool):
    name = "transform_text"
    description = "Transform or summarize provided text according to an instruction. Use for safe in-context text processing before writing files."
    argument_schema = TransformTextArguments
    risk_level = RiskLevel.READ_ONLY

    def execute(self, arguments: TransformTextArguments) -> str:
        raise RuntimeError("transform_text requires an LLM-backed executor implementation")

