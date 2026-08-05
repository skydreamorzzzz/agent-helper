from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MemoryCategory(StrEnum):
    USER_PREFERENCE = "user_preference"
    PERSONAL_FACT = "personal_fact"
    PROJECT = "project"
    TASK = "task"
    DECISION = "decision"
    SUMMARY = "summary"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: int
    category: MemoryCategory
    content: str
    source_run_id: str
    source_message_id: str
    importance: float = Field(ge=0.0, le=1.0)
    created_at: str
    updated_at: str
    last_accessed_at: str | None = None
    is_active: bool = True
    metadata_json: str = "{}"


class MemoryCreate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    category: MemoryCategory = MemoryCategory.PERSONAL_FACT
    content: str = Field(min_length=1, max_length=4000)
    source_run_id: str
    source_message_id: str
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class RetrievedMemory(BaseModel):
    memory: Memory
    score: float


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    category: MemoryCategory
    content: str = Field(min_length=1, max_length=4000)
    importance: float = Field(ge=0.0, le=1.0)
    reason: str = ""

