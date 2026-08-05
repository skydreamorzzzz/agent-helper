from __future__ import annotations

from difflib import SequenceMatcher
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from src.agent.protocol import ModelOutputParseError, extract_first_json_object
from src.memory.models import MemoryCandidate, MemoryCategory, MemoryCreate
from src.memory.store import MemoryStore


class CandidateExtractionResult(BaseModel):
    candidates: list[MemoryCandidate] = Field(default_factory=list)


class CandidateLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class MemoryWriter:
    def __init__(
        self,
        store: MemoryStore,
        *,
        importance_threshold: float = 0.6,
        duplicate_threshold: float = 0.88,
    ) -> None:
        self.store = store
        self.importance_threshold = importance_threshold
        self.duplicate_threshold = duplicate_threshold

    def remember_explicit(
        self,
        content: str,
        *,
        source_run_id: str,
        source_message_id: str,
        category: MemoryCategory = MemoryCategory.PROJECT,
        importance: float = 0.8,
    ):
        return self.store.add(
            MemoryCreate(
                category=category,
                content=content,
                source_run_id=source_run_id,
                source_message_id=source_message_id,
                importance=importance,
                metadata={"write_mode": "explicit"},
            )
        )

    def save_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        source_run_id: str,
        source_message_id: str,
    ):
        allowed, reason = self.should_save(candidate)
        if not allowed:
            return None, reason
        memory = self.store.add(
            MemoryCreate(
                category=candidate.category,
                content=candidate.content,
                source_run_id=source_run_id,
                source_message_id=source_message_id,
                importance=candidate.importance,
                metadata={"write_mode": "candidate", "reason": candidate.reason},
            )
        )
        return memory, "saved"

    def should_save(self, candidate: MemoryCandidate) -> tuple[bool, str]:
        content = candidate.content.strip()
        if candidate.importance < self.importance_threshold:
            return False, "importance_below_threshold"
        if len(content) < 8:
            return False, "too_short"
        if self._looks_temporary(content):
            return False, "temporary_content"
        if self._looks_like_guess(content):
            return False, "possible_model_guess"
        if self.is_duplicate(content):
            return False, "duplicate"
        return True, "accepted"

    def is_duplicate(self, content: str) -> bool:
        normalized = self._normalize(content)
        for memory in self.store.list_active():
            existing = self._normalize(memory.content)
            if normalized == existing:
                return True
            if SequenceMatcher(None, normalized, existing).ratio() >= self.duplicate_threshold:
                return True
        return False

    def extract_candidates(
        self,
        llm_client: CandidateLLM,
        *,
        conversation_text: str,
    ) -> list[MemoryCandidate]:
        prompt = (
            "Extract only durable, user-grounded long-term memory candidates.\n"
            "Do not include small talk, temporary facts, guesses, or sensitive/uncertain facts.\n"
            "Return JSON only: {\"candidates\":[{\"category\":\"project\",\"content\":\"...\",\"importance\":0.8,\"reason\":\"...\"}]}\n"
            f"Conversation:\n{conversation_text}"
        )
        raw = llm_client.chat([{"role": "user", "content": prompt}])
        json_text = extract_first_json_object(raw)
        if json_text is None:
            raise ModelOutputParseError("Candidate extraction returned no JSON object")
        try:
            return CandidateExtractionResult.model_validate_json(json_text).candidates
        except ValidationError as exc:
            raise ModelOutputParseError(str(exc)) from exc

    def _looks_temporary(self, content: str) -> bool:
        lowered = content.lower()
        temporary_markers = ("今天", "明天", "刚才", "临时", "这次", "现在时间", "today", "tomorrow", "temporary")
        return any(marker in lowered for marker in temporary_markers)

    def _looks_like_guess(self, content: str) -> bool:
        lowered = content.lower()
        guess_markers = ("可能", "也许", "大概", "推测", "猜测", "probably", "maybe", "might")
        return any(marker in lowered for marker in guess_markers)

    def _normalize(self, content: str) -> str:
        return " ".join(content.lower().split())

