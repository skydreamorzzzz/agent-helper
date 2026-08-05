from __future__ import annotations

from pathlib import Path

from src.memory.journal import DevelopmentJournal
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.planner.models import ExecutionResult, Plan, PlanStep, StepStatus


def test_development_journal_records_valuable_issue_in_chinese(tmp_path: Path) -> None:
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"))
    plan = Plan(
        goal="整理文件并写入结果",
        steps=[
            PlanStep(id="s1", description="读", tool_name="read_text_file", arguments={"path": "todo.txt"}, status=StepStatus.COMPLETED),
            PlanStep(id="s2", description="整理", tool_name="transform_text", arguments={"input_text": "x", "instruction": "整理"}, status=StepStatus.COMPLETED),
            PlanStep(id="s3", description="写", tool_name="write_text_file", arguments={"path": "out.md", "content": "x"}, status=StepStatus.COMPLETED),
        ],
    )
    result = ExecutionResult(plan=plan, final_answer="done", stopped_reason="completed")

    journal_path = tmp_path / "development_journal.md"
    outcome = DevelopmentJournal(service, markdown_path=journal_path).record_after_plan(result, source_run_id="run-1")

    assert outcome.markdown_path == str(journal_path)
    contents = journal_path.read_text(encoding="utf-8")
    assert "read_text_file -> transform_text -> write_text_file" in contents
    assert service.list_memories() == []


def test_user_journal_requirement_can_be_remembered(tmp_path: Path) -> None:
    service = MemoryService(store=MemoryStore(tmp_path / "memory.sqlite3"))

    memory = service.remember_explicit(
        "项目要求：每次完成开发或复杂任务后，先判断过程中是否出现有价值的问题；如果问题有助于解释项目设计、架构取舍、失败恢复或简历答辩，就用中文记录到长期记忆，不要记录普通流水账。",
        source_run_id="run-2",
    )

    assert "用中文记录到长期记忆" in memory.content
