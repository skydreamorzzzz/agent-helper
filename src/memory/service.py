from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Protocol

from src.memory.models import Memory, MemoryCategory, RetrievedMemory
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.memory.summarizer import ConversationSummarizer
from src.memory.writer import MemoryWriter


class MemoryLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass
class MemoryContext:
    retrieved: list[RetrievedMemory]
    block: str


class MemoryService:
    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        retriever: MemoryRetriever | None = None,
        writer: MemoryWriter | None = None,
        summarizer: ConversationSummarizer | None = None,
        retrieval_limit: int = 5,
    ) -> None:
        self.store = store or MemoryStore()
        self.retriever = retriever or MemoryRetriever(self.store)
        self.writer = writer or MemoryWriter(self.store)
        self.summarizer = summarizer or ConversationSummarizer()
        self.retrieval_limit = retrieval_limit
        self.conversation_summary = ""
        self.recent_messages: list[dict[str, str]] = []

    def remember_explicit(self, content: str, *, source_run_id: str | None = None) -> Memory:
        return self.writer.remember_explicit(
            content,
            source_run_id=source_run_id or f"manual-{uuid.uuid4().hex}",
            source_message_id="cli:/remember",
            category=MemoryCategory.PROJECT,
            importance=0.85,
        )

    def list_memories(self) -> list[Memory]:
        return self.store.list_active()

    def forget(self, memory_id: int) -> None:
        self.store.soft_delete(memory_id)

    def update_memory(self, memory_id: int, content: str) -> Memory:
        return self.store.update(memory_id, content)

    def extract_and_save_candidates(
        self,
        llm_client: MemoryLLM,
        *,
        conversation_text: str,
        source_run_id: str,
        source_message_id: str,
    ) -> list[tuple[Memory | None, str]]:
        candidates = self.writer.extract_candidates(llm_client, conversation_text=conversation_text)
        return [
            self.writer.save_candidate(
                candidate,
                source_run_id=source_run_id,
                source_message_id=source_message_id,
            )
            for candidate in candidates
        ]

    def retrieve_context(self, query: str, *, limit: int | None = None) -> MemoryContext:
        retrieved = self.retriever.search(query, limit=limit or self.retrieval_limit)
        return MemoryContext(retrieved=retrieved, block=self.format_memory_context(retrieved))

    def format_memory_context(self, memories: list[RetrievedMemory]) -> str:
        if not memories:
            return "No relevant long-term memories retrieved."
        lines = [
            "Long-term memories may be outdated or incomplete. Treat them as context, not absolute facts.",
            "If the current user request conflicts with old memories, follow the current user request.",
        ]
        for item in memories:
            memory = item.memory
            lines.append(
                f"- id={memory.id}; category={memory.category}; importance={memory.importance:.2f}; "
                f"score={item.score:.4f}; source_run_id={memory.source_run_id}; "
                f"content={memory.content}"
            )
        return "\n".join(lines)

    def append_recent_turn(
        self,
        user_input: str,
        assistant_output: str,
        *,
        max_messages: int,
        max_chars: int,
        summary_trigger_messages: int,
    ) -> None:
        self.recent_messages.append({"role": "user", "content": user_input})
        self.recent_messages.append({"role": "assistant", "content": assistant_output})
        if self.summarizer.should_summarize(self.recent_messages, summary_trigger_messages):
            messages_to_summarize = self.recent_messages[:-max_messages] if max_messages else self.recent_messages
            if not messages_to_summarize:
                messages_to_summarize = self.recent_messages
            self.conversation_summary = self.summarizer.summarize(
                existing_summary=self.conversation_summary,
                messages=messages_to_summarize,
            )
            self.recent_messages = self.recent_messages[-max_messages:]
        self.truncate_working_memory(max_messages=max_messages, max_chars=max_chars)

    def truncate_working_memory(self, *, max_messages: int, max_chars: int) -> None:
        if max_messages > 0:
            self.recent_messages = self.recent_messages[-max_messages:]
        while self._message_chars() > max_chars and self.recent_messages:
            self.recent_messages.pop(0)

    def clear_session(self) -> None:
        self.recent_messages.clear()
        self.conversation_summary = ""

    def _message_chars(self) -> int:
        return sum(len(message["content"]) for message in self.recent_messages)
