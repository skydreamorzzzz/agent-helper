from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.memory.models import utc_now_iso


class Route(StrEnum):
    DIRECT_ANSWER = "direct_answer"
    SINGLE_TOOL = "single_tool"
    WEB_LOOKUP = "web_lookup"
    PLANNED_TASK = "planned_task"
    DEEP_RESEARCH = "deep_research"
    CLARIFICATION = "clarification"


class RouteDecision(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    route: Route
    reason: str
    missing_information: list[str] = Field(default_factory=list)


class PlanStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"


class PlanStep(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str
    description: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    expected_output: str = ""
    actual_output: Any | None = None
    error: str | None = None
    retry_count: int = 0


class ReplanRecord(BaseModel):
    reason: str
    created_at: str = Field(default_factory=utc_now_iso)
    original_plan_json: str


class Plan(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    plan_id: str = Field(default_factory=lambda: uuid4().hex)
    goal: str
    status: PlanStatus = PlanStatus.PENDING
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    current_step_id: str | None = None
    steps: list[PlanStep]
    assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    final_output_requirement: str = ""
    replan_count: int = 0
    replan_history: list[ReplanRecord] = Field(default_factory=list)


class PlanValidationResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    plan: Plan
    final_answer: str
    stopped_reason: Literal[
        "completed",
        "failed",
        "cancelled",
        "confirmation_required",
        "replan_limit_reached",
    ]
