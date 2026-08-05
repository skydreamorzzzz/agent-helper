from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.memory.models import Memory, MemoryCandidate, MemoryCategory
from src.memory.service import MemoryService
from src.planner.models import ExecutionResult, PlanStatus, StepStatus


@dataclass(frozen=True)
class JournalOutcome:
    saved: list[Memory]
    skipped_reasons: list[str]
    markdown_path: str | None = None


class DevelopmentJournal:
    def __init__(self, memory_service: MemoryService, *, markdown_path: Path = Path("docs/development_journal.md")) -> None:
        self.memory_service = memory_service
        self.markdown_path = markdown_path

    def record_after_plan(
        self,
        result: ExecutionResult,
        *,
        source_run_id: str,
    ) -> JournalOutcome:
        candidates = self._build_candidates(result)
        saved: list[Memory] = []
        skipped: list[str] = []
        entries = [candidate for candidate in candidates if not self.memory_service.writer.is_duplicate(candidate.content)]
        if entries:
            self._append_markdown_entries(result, entries, source_run_id=source_run_id)
        else:
            skipped.append("no_valuable_journal_entry")
        return JournalOutcome(
            saved=saved,
            skipped_reasons=skipped,
            markdown_path=str(self.markdown_path) if entries else None,
        )

    def _build_candidates(self, result: ExecutionResult) -> list[MemoryCandidate]:
        plan = result.plan
        candidates: list[MemoryCandidate] = []

        if any(step.tool_name == "transform_text" for step in plan.steps):
            candidates.append(
                MemoryCandidate(
                    category=MemoryCategory.DECISION,
                    content=(
                        "项目经验记录：复杂任务中，读取文件后需要整理、抽取或总结内容时，"
                        "应使用 read_text_file -> transform_text -> write_text_file 的三步模式，"
                        "不要让 write_text_file 承担文本理解或抽取逻辑。"
                    ),
                    importance=0.82,
                    reason="这是一次真实全链路测试中暴露出的工具职责边界问题，适合简历项目答辩时说明架构演进。",
                )
            )

        failed_steps = [step for step in plan.steps if step.status == StepStatus.FAILED]
        if failed_steps or result.stopped_reason in {"failed", "replan_limit_reached"}:
            errors = "; ".join(step.error or step.id for step in failed_steps[:3])
            candidates.append(
                MemoryCandidate(
                    category=MemoryCategory.PROJECT,
                    content=(
                        "项目经验记录：计划执行失败时，需要记录失败步骤、工具参数、错误原因和是否触发重试或重规划；"
                        f"本次失败摘要：{errors or result.final_answer}"
                    ),
                    importance=0.78,
                    reason="失败恢复是该 Agent 项目的核心能力，失败样例有助于后续复盘。",
                )
            )

        if plan.status == PlanStatus.COMPLETED and result.stopped_reason == "completed":
            candidates.append(
                MemoryCandidate(
                    category=MemoryCategory.PROJECT,
                    content=(
                        "项目要求：每次完成开发或复杂任务后，先判断过程中是否出现有价值的问题；"
                        "如果问题有助于解释项目设计、架构取舍、失败恢复或简历答辩，就用中文记录到长期记忆，"
                        "不要记录普通流水账。"
                    ),
                    importance=0.9,
                    reason="这是用户明确提出的项目级长期要求。",
                )
            )

        return candidates

    def _append_markdown_entries(
        self,
        result: ExecutionResult,
        entries: list[MemoryCandidate],
        *,
        source_run_id: str,
    ) -> None:
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# 开发复盘记录\n\n", encoding="utf-8")

        sections = []
        for entry in entries:
            sections.append(
                "\n".join(
                    [
                        "## 复盘条目",
                        "",
                        f"- 来源 run_id：`{source_run_id}`",
                        f"- 计划 ID：`{result.plan.plan_id}`",
                        f"- 分类：`{entry.category}`",
                        f"- 重要性：`{entry.importance}`",
                        f"- 原因：{entry.reason}",
                        "",
                        entry.content,
                        "",
                    ]
                )
            )
        with self.markdown_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(sections))
