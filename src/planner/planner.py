from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from src.agent.protocol import ModelOutputParseError, extract_first_json_object
from src.planner.models import Plan, PlanStep
from src.planner.prompts import build_planner_prompt
from src.tools.registry import ToolRegistry


class PlannerLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class RawPlan(BaseModel):
    goal: str
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    final_output_requirement: str = ""


class StructuredPlanner:
    def __init__(self, llm_client: PlannerLLM, registry: ToolRegistry, *, max_steps: int = 8) -> None:
        self.llm_client = llm_client
        self.registry = registry
        self.max_steps = max_steps

    def create_plan(self, user_input: str, *, memory_context: str = "") -> Plan:
        raw = self.llm_client.chat(
            [
                {
                    "role": "user",
                    "content": build_planner_prompt(
                        user_input=user_input,
                        registry=self.registry,
                        memory_context=memory_context,
                        max_steps=self.max_steps,
                    ),
                }
            ]
        )
        json_text = extract_first_json_object(raw)
        if json_text is None:
            raise ModelOutputParseError("Planner returned no JSON object")
        try:
            raw_plan = RawPlan.model_validate_json(json_text)
        except ValidationError as exc:
            raise ModelOutputParseError(str(exc)) from exc
        return Plan(
            goal=raw_plan.goal,
            steps=raw_plan.steps,
            assumptions=raw_plan.assumptions,
            unresolved_questions=raw_plan.unresolved_questions,
            final_output_requirement=raw_plan.final_output_requirement,
        )

