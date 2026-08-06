from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.runtime import AgentRuntime
from src.cli import build_research_registry
from src.config import LOGS_DIR
from src.llm.client import LocalLLMClient
from src.planner.executor import PlanExecutor
from src.planner.models import Route, StepStatus
from src.planner.planner import StructuredPlanner
from src.planner.repository import PlanRepository
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.registry import ToolRegistry

import src.planner.executor as executor_module

from evals.gaia import GaiaQuestion, is_correct

EXTRACTION_PROMPT = """Answer the question by giving the EXACT short answer only.
The question and a candidate response from an assistant are provided below.
Return ONLY the precise answer to the question: a number, a short phrase, or a comma-separated list.
Do not include explanations, quotes, or the word "answer". If no answer can be determined, return the single word: None.

Question:
{question}

Candidate response:
{candidate}
"""


class CountingLLM:
    def __init__(self, inner: LocalLLMClient) -> None:
        self.inner = inner
        self.count = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.count += 1
        return self.inner.chat(messages)


@dataclass
class QuestionRecord:
    task_id: str
    question: str
    level: int
    expected_answer: str
    route: str = ""
    stopped_reason: str = ""
    search_count: int = 0
    search_queries: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    report_file: str | None = None
    agent_output: str | None = None
    extracted_answer: str | None = None
    passed: bool = False
    llm_calls: int = 0
    search_credits: int = 0
    citation_total: int = 0
    citation_ok: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "level": self.level,
            "expected_answer": self.expected_answer,
            "route": self.route,
            "stopped_reason": self.stopped_reason,
            "search_count": self.search_count,
            "search_queries": self.search_queries,
            "tool_names": self.tool_names,
            "validation_errors": self.validation_errors,
            "report_file": self.report_file,
            "agent_output": self.agent_output,
            "extracted_answer": self.extracted_answer,
            "passed": self.passed,
            "llm_calls": self.llm_calls,
            "search_credits": self.search_credits,
            "citation_total": self.citation_total,
            "citation_ok": self.citation_ok,
        }


def _resolve_in(root: Path, p: str) -> Path:
    root_r = root.resolve()
    candidate = (root / p).resolve()
    if candidate != root_r and root_r not in candidate.parents:
        raise ValueError("Path traversal outside workspace is not allowed")
    if candidate == root_r:
        raise ValueError("Path must refer to a file inside workspace")
    return candidate


def _read_runtime_trajectory(run_id: str) -> tuple[list[str], list[str]]:
    log_path = LOGS_DIR / f"{run_id}.log"
    tool_names: list[str] = []
    search_queries: list[str] = []
    if not log_path.exists():
        return tool_names, search_queries
    for line in log_path.read_text(encoding="utf-8").splitlines():
        json_start = line.find("{")
        if json_start >= 0:
            line = line[json_start:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "tool_call":
            tool_names.append(str(event.get("tool", "")))
            if event.get("tool") == "search_web":
                args = event.get("arguments") or {}
                search_queries.append(str(args.get("query", "")))
    return tool_names, search_queries


def _check_citations(plan, report_path: Path) -> tuple[int, int]:
    material_urls = set()
    for step in plan.steps:
        if step.tool_name == "search_web" and step.status == StepStatus.COMPLETED and step.actual_output:
            try:
                results = json.loads(step.actual_output) if isinstance(step.actual_output, str) else step.actual_output
            except (json.JSONDecodeError, TypeError):
                continue
            for item in results:
                url = (item or {}).get("url")
                if url:
                    material_urls.add(str(url).rstrip("/"))
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    urls = {u.rstrip("/") for u in re.findall(r"https?://[^\s)\]]+", text)}
    total = len(urls)
    ok = sum(1 for u in urls if u in material_urls)
    return total, ok


def _extract_answer(llm, question: str, candidate: str) -> str:
    raw = llm.chat([{"role": "user", "content": EXTRACTION_PROMPT.format(question=question, candidate=candidate)}]).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw).strip()
    for label in ("Answer:", "answer:", "答案是：", "答案："):
        if raw.startswith(label):
            raw = raw[len(label):].strip()
    if len(raw) >= 2 and raw[0] in "\"'“”" and raw[-1] == raw[0]:
        raw = raw[1:-1].strip()
    return raw


def _run_research(rec: QuestionRecord, q: GaiaQuestion, counting, research_registry, settings, repo, validator, research_planner, q_dir: Path) -> None:
    try:
        plan = research_planner.build_research_plan(q.question, memory_context="")
    except Exception as exc:
        rec.stopped_reason = f"plan_error:{type(exc).__name__}"
        return
    validation = validator.validate(plan)
    rec.validation_errors = list(validation.errors)
    if not validation.ok:
        rec.stopped_reason = "validation_failed"
        return

    def replan_callback(old_plan, reason):
        topic = old_plan.goal.removeprefix("深度调研：")
        try:
            return research_planner.build_research_plan(f"请重新调研 {topic}", memory_context="")
        except Exception:
            return old_plan

    result = PlanExecutor(
        registry=research_registry,
        repository=repo,
        llm_client=counting,
        validator=validator,
        max_retries=settings.tool_max_retries,
        max_replans=settings.planner_max_replans,
        confirm_write_actions=False,
        confirmation_callback=lambda *a: True,
        replan_callback=replan_callback,
        logger=None,
    ).execute(plan)
    rec.agent_output = result.final_answer
    rec.stopped_reason = result.stopped_reason
    for step in result.plan.steps:
        rec.tool_names.append(step.tool_name)
        if step.tool_name == "search_web":
            rec.search_count += 1
            rec.search_queries.append(str(step.arguments.get("query", "")))
            if step.status == StepStatus.COMPLETED:
                rec.search_credits += 1
        elif step.tool_name == "write_cited_report":
            rec.report_file = str(step.arguments.get("report_file", ""))
    if rec.report_file:
        report_path = _resolve_in(q_dir, rec.report_file)
        rec.citation_total, rec.citation_ok = _check_citations(result.plan, report_path)


def _run_runtime(rec: QuestionRecord, q: GaiaQuestion, runtime: AgentRuntime) -> None:
    result = runtime.run(q.question)
    rec.agent_output = result.content
    rec.stopped_reason = result.stopped_reason
    tool_names, queries = _read_runtime_trajectory(result.run_id)
    rec.tool_names = tool_names
    rec.search_queries = queries
    rec.search_count = len(queries)
    rec.search_credits = len(queries)


def run_question(
    q: GaiaQuestion,
    *,
    llm_client: LocalLLMClient,
    research_registry: ToolRegistry,
    settings,
    workspace_root: Path,
) -> QuestionRecord:
    rec = QuestionRecord(task_id=q.task_id, question=q.question, level=q.level, expected_answer=q.answer)
    q_dir = workspace_root / q.task_id
    q_dir.mkdir(parents=True, exist_ok=True)
    executor_module.resolve_workspace_path = lambda p: _resolve_in(q_dir, p)

    repo = PlanRepository(q_dir / "plans.sqlite3")
    counting = CountingLLM(llm_client)
    router = RequestRouter(counting)
    research_planner = StructuredPlanner(counting, research_registry, max_steps=settings.planner_max_steps)
    validator = PlanValidator(research_registry, max_steps=settings.planner_max_steps)
    runtime = AgentRuntime(
        llm_client=counting,
        tool_registry=research_registry,
        max_tool_calls=settings.max_tool_calls,
        memory_service=None,
    )

    decision = router.route(q.question)
    rec.route = decision.route

    if decision.route == Route.DEEP_RESEARCH:
        _run_research(rec, q, counting, research_registry, settings, repo, validator, research_planner, q_dir)
    else:
        _run_runtime(rec, q, runtime)

    rec.llm_calls = counting.count
    if rec.agent_output:
        try:
            rec.extracted_answer = _extract_answer(counting, q.question, rec.agent_output)
        except Exception:
            rec.extracted_answer = None
        rec.passed = is_correct(rec.extracted_answer or "", q.answer)
    return rec
