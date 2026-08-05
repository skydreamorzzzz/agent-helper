from __future__ import annotations

from pathlib import Path

from src.memory.models import MemoryCandidate, MemoryCategory, MemoryCreate
from src.memory.retriever import MemoryRetriever
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.memory.summarizer import ConversationSummarizer
from src.memory.writer import MemoryWriter
from src.tools.calculator import CalculatorTool
from src.tools.registry import ToolRegistry
from src.agent.prompts import build_system_prompt


def make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory.sqlite3")


def add_memory(store: MemoryStore, content: str, *, category: MemoryCategory = MemoryCategory.PROJECT):
    return store.add(
        MemoryCreate(
            category=category,
            content=content,
            source_run_id="run-1",
            source_message_id="message-1",
            importance=0.8,
        )
    )


def test_save_and_read_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)

    saved = add_memory(store, "用户正在开发本地个人助手项目")
    loaded = store.get(saved.id)

    assert loaded is not None
    assert loaded.content == "用户正在开发本地个人助手项目"
    assert loaded.category == MemoryCategory.PROJECT


def test_soft_delete_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    saved = add_memory(store, "delete me")

    store.soft_delete(saved.id)

    assert store.get(saved.id) is None
    inactive = store.get(saved.id, include_inactive=True)
    assert inactive is not None
    assert inactive.is_active is False


def test_update_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    saved = add_memory(store, "old content")

    updated = store.update(saved.id, "new content")

    assert updated.content == "new content"


def test_fts5_retrieval(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    add_memory(store, "The assistant project focuses on planning memory and tools")
    retriever = MemoryRetriever(store)

    results = retriever.search("memory tools", limit=3)

    assert results
    assert "planning memory and tools" in results[0].memory.content


def test_retrieval_does_not_return_deleted_memory(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    saved = add_memory(store, "The assistant project focuses on memory")
    store.soft_delete(saved.id)
    retriever = MemoryRetriever(store)

    assert retriever.search("memory", limit=3) == []


def test_duplicate_memory_detection(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    add_memory(store, "用户正在开发本地个人助手项目")
    writer = MemoryWriter(store)

    assert writer.is_duplicate("用户正在开发本地个人助手项目")
    candidate = MemoryCandidate(
        category=MemoryCategory.PROJECT,
        content="用户正在开发本地个人助手项目",
        importance=0.9,
    )
    assert writer.should_save(candidate) == (False, "duplicate")


def test_working_memory_truncation(tmp_path: Path) -> None:
    service = MemoryService(store=make_store(tmp_path))
    service.recent_messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]

    service.truncate_working_memory(max_messages=2, max_chars=100)

    assert [message["content"] for message in service.recent_messages] == ["three", "four"]


def test_conversation_summary_trigger(tmp_path: Path) -> None:
    service = MemoryService(store=make_store(tmp_path), summarizer=ConversationSummarizer())

    service.append_recent_turn(
        "项目必须支持记忆",
        "已记录要求",
        max_messages=2,
        max_chars=1000,
        summary_trigger_messages=1,
    )

    assert "项目必须支持记忆" in service.conversation_summary
    assert len(service.recent_messages) == 2


def test_memory_injection_format(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    saved = add_memory(store, "项目重点是规划、记忆和工具")
    service = MemoryService(store=store, retriever=MemoryRetriever(store))

    context = service.retrieve_context("个人助手项目重点", limit=3)

    assert f"id={saved.id}" in context.block
    assert "source_run_id=run-1" in context.block
    assert "可能过时" not in context.block
    assert "outdated" in context.block


def test_current_user_request_overrides_old_memory_instruction(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    prompt = build_system_prompt(
        registry,
        memory_context="Old memory says the project focus is tools only.",
        conversation_summary="No summary.",
    )

    assert "If the current user request conflicts with old memories" in prompt
    assert "Current User Request" not in prompt

