from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.runtime import AgentRuntime
from src.cli import build_lookup_registry, build_registry, build_research_registry
from src.config import LOGS_DIR, load_settings
from src.llm.client import LocalLLMClient
from src.logging_utils import log_event, setup_run_logger
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.planner.executor import PlanExecutor
from src.planner.models import Plan, PlanStatus, RiskLevel, Route, StepStatus
from src.planner.planner import StructuredPlanner
from src.planner.repository import PlanRepository
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.registry import ToolRegistry

from evals.agent.evaluators import evaluate_task
from evals.agent.report import dataset_fingerprint, git_commit, git_dirty, write_outputs
from evals.agent.semantics import tool_event_summary

import src.planner.executor as executor_module
import src.tools.file_tools as file_tools_module


DATASET_PATH = Path(__file__).with_name("live_dataset.jsonl")
RESULTS_ROOT = Path(__file__).with_name("results")
DATASET_VERSION = "agent-live-e2e-v1.1"


@dataclass
class CountingLLMClient:
    client: LocalLLMClient
    count: int = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.count += 1
        return self.client.chat(messages)


@dataclass
class CaseOutcome:
    record: dict[str, Any]
    workspace_dir: Path


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if item["id"] in seen:
                raise ValueError(f"duplicate dataset id at line {line_no}: {item['id']}")
            seen.add(item["id"])
            examples.append(item)
    return examples


def run_case(example: dict[str, Any], *, root_dir: Path, llm: CountingLLMClient, settings) -> CaseOutcome:
    workspace_dir = root_dir / example["id"] / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _write_setup_files(workspace_dir, example.get("setup") or {})
    memory_service = _build_memory_service(example, workspace_dir, settings)
    record = _empty_record(example)
    start_llm_calls = llm.count
    start = time.perf_counter()

    with _patched_workspace(workspace_dir):
        try:
            record.update(_execute_pipeline(example, llm, settings, workspace_dir, memory_service))
        except Exception as exc:
            record["final_status"] = "runner_error"
            record["stopped_reason"] = f"runner_error:{type(exc).__name__}: {exc}"

    record["llm_calls"] = llm.count - start_llm_calls
    record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    evaluation = evaluate_task(example, record, workspace_dir)
    record["task_success"] = evaluation.task_success
    record["execution_contract_pass"] = evaluation.execution_contract_pass
    record["route_correct"] = evaluation.route_correct
    record["failure_reasons"] = evaluation.failure_reasons
    record["task_failure_reasons"] = evaluation.task_failure_reasons
    record["execution_failure_reasons"] = evaluation.execution_failure_reasons
    record["failure_stage"] = evaluation.failure_stage
    record["execution_failure_stage"] = evaluation.execution_failure_stage
    return CaseOutcome(record=record, workspace_dir=workspace_dir)


def _execute_pipeline(
    example: dict[str, Any],
    llm: CountingLLMClient,
    settings,
    workspace_dir: Path,
    memory_service: MemoryService | None,
) -> dict[str, Any]:
    core_registry = _observed_registry(build_registry())
    lookup_registry = _observed_registry(build_lookup_registry(settings.tavily_api_key))
    research_registry = _observed_registry(build_research_registry(settings.tavily_api_key))
    router = RequestRouter(llm)
    memory_context = memory_service.retrieve_context(example["input"], limit=settings.memory_retrieval_limit).block if memory_service else ""
    decision = router.route(example["input"], memory_context=memory_context)
    record = {"actual_route": str(decision.route)}

    if decision.route == Route.CLARIFICATION:
        record.update(
            {
                "final_status": "clarification",
                "stopped_reason": "clarification",
                "final_answer": " ".join(decision.missing_information),
                **_tool_event_summary([]),
            }
        )
        return record

    if decision.route == Route.WEB_LOOKUP:
        if not settings.tavily_api_key:
            record.update(
                {
                    "final_status": "missing_tavily_api_key",
                    "stopped_reason": "missing_tavily_api_key",
                    "final_answer": "未配置 TAVILY_API_KEY，无法执行联网查询。",
                    **_tool_event_summary([]),
                    "retry_count": 0,
                    "replan_count": 0,
                }
            )
            return record
        result = AgentRuntime(
            llm_client=llm,
            tool_registry=lookup_registry,
            max_tool_calls=settings.max_tool_calls,
            memory_service=memory_service,
            working_memory_max_messages=settings.working_memory_max_messages,
            working_memory_max_chars=settings.working_memory_max_chars,
            summary_trigger_messages=settings.summary_trigger_messages,
            confirm_write_actions=settings.confirm_write_actions,
            confirmation_callback=_runtime_confirmation(example),
        ).run(
            example["input"],
            required_tool="search_web",
            execution_policy="This request was routed as web_lookup. You must call search_web before final_answer.",
        )
        record.update(_runtime_record(result.run_id, result.content, result.stopped_reason))
        return record

    if decision.route in (Route.PLANNED_TASK, Route.DEEP_RESEARCH):
        is_research = decision.route == Route.DEEP_RESEARCH
        if is_research and not settings.tavily_api_key:
            record.update(
                {
                    "final_status": "missing_tavily_api_key",
                    "stopped_reason": "missing_tavily_api_key",
                    "final_answer": "未配置 TAVILY_API_KEY，无法执行深度调研。",
                    **_tool_event_summary([]),
                    "retry_count": 0,
                    "replan_count": 0,
                }
            )
            return record
        registry = research_registry if is_research else core_registry
        planner = StructuredPlanner(llm, registry, max_steps=settings.planner_max_steps)
        repo = PlanRepository(workspace_dir / "plans.sqlite3")
        validator = PlanValidator(registry, max_steps=settings.planner_max_steps)
        run_id = uuid.uuid4().hex
        logger = setup_run_logger(run_id, LOGS_DIR)
        log_event(logger, "request_route", decision=decision.model_dump())
        try:
            plan = (
                planner.build_research_plan(example["input"], memory_context=memory_context)
                if is_research
                else planner.create_plan(example["input"], memory_context=memory_context)
            )
        except Exception as exc:
            record.update(
                {
                    "final_status": "planning_failed",
                    "stopped_reason": f"planning_failed:{type(exc).__name__}: {exc}",
                    **_tool_event_summary(_registry_events(registry)),
                    "retry_count": 0,
                    "replan_count": 0,
                }
            )
            return record
        if is_research and not bool(example.get("confirmation", True)):
            plan.status = PlanStatus.CANCELLED
            repo.save(plan)
            record.update(
                {
                    "final_status": "cancelled",
                    "stopped_reason": "cancelled",
                    "final_answer": "已取消本次深度调研。",
                    **_tool_event_summary([]),
                    "retry_count": 0,
                    "replan_count": 0,
                }
            )
            return record
        result = PlanExecutor(
            registry=registry,
            repository=repo,
            llm_client=llm,
            validator=validator,
            max_retries=settings.tool_max_retries,
            max_replans=settings.planner_max_replans,
            confirm_write_actions=settings.confirm_write_actions,
            confirmation_callback=_plan_confirmation(example),
            replan_callback=_replan_callback(example, planner) if is_research else None,
            logger=logger,
        ).execute(plan)
        record.update(_plan_record(result.plan, result.final_answer, result.stopped_reason, _registry_events(registry)))
        return record

    result = AgentRuntime(
        llm_client=llm,
        tool_registry=core_registry,
        max_tool_calls=settings.max_tool_calls,
        memory_service=memory_service,
        working_memory_max_messages=settings.working_memory_max_messages,
        working_memory_max_chars=settings.working_memory_max_chars,
        summary_trigger_messages=settings.summary_trigger_messages,
        confirm_write_actions=settings.confirm_write_actions,
        confirmation_callback=_runtime_confirmation(example),
    ).run(example["input"])
    record.update(_runtime_record(result.run_id, result.content, result.stopped_reason))
    return record


def _runtime_record(run_id: str, final_answer: str, stopped_reason: str) -> dict[str, Any]:
    events = _events_from_log(LOGS_DIR / f"{run_id}.log")
    return {
        "final_status": stopped_reason,
        "stopped_reason": stopped_reason,
        "final_answer": final_answer,
        **_tool_event_summary(events),
        "retry_count": 1 if any(event.get("event") == "model_repair_output" for event in events) else 0,
        "replan_count": 0,
    }


def _plan_record(plan: Plan, final_answer: str, stopped_reason: str, registry_events: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for step in plan.steps:
        if step.status in (StepStatus.COMPLETED, StepStatus.FAILED):
            events.append({"tool": step.tool_name, "event": "planned_tool_step", "source": "plan", "step_id": step.id})
            events.append({"tool": step.tool_name, "event": "execution_attempt", "step_id": step.id})
            events.append(
                {
                    "tool": step.tool_name,
                    "event": "execution_success" if step.status == StepStatus.COMPLETED else "execution_failure",
                    "step_id": step.id,
                }
            )
    if stopped_reason == "confirmation_required" and plan.current_step_id:
        current = next((step for step in plan.steps if step.id == plan.current_step_id), None)
        if current is not None:
            events.append({"tool": current.tool_name, "event": "planned_tool_step", "source": "plan", "step_id": current.id})
            events.append({"tool": current.tool_name, "event": "policy_rejected", "source": "plan_policy", "step_id": current.id})
    registry_keys = {(event.get("tool"), event.get("event")) for event in registry_events}
    existing_keys = {(event.get("tool"), event.get("event")) for event in events}
    events.extend(event for event in registry_events if (event.get("tool"), event.get("event")) not in existing_keys and (event.get("tool"), event.get("event")) in registry_keys)
    return {
        "final_status": stopped_reason,
        "stopped_reason": stopped_reason,
        "final_answer": final_answer,
        **_tool_event_summary(events),
        "retry_count": sum(step.retry_count for step in plan.steps),
        "replan_count": plan.replan_count,
    }


def _events_from_log(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.find("{")
        if index < 0:
            continue
        try:
            payload = json.loads(line[index:])
        except json.JSONDecodeError:
            continue
        event_name = payload.get("event")
        tool = payload.get("tool")
        if event_name == "tool_call" and tool:
            events.append({"tool": tool, "event": "model_tool_proposed", "source": "model"})
        elif event_name == "tool_result" and tool:
            result = payload.get("result") or {}
            if result.get("ok"):
                events.append({"tool": tool, "event": "execution_attempt"})
                events.append({"tool": tool, "event": "execution_success"})
            elif "requires confirmation" in str(result.get("error") or "") or "denied" in str(result.get("error") or ""):
                events.append({"tool": tool, "event": "policy_rejected", "source": "runtime_policy"})
            else:
                events.append({"tool": tool, "event": "execution_attempt"})
                events.append({"tool": tool, "event": "execution_failure", "error": result.get("error")})
        elif event_name == "model_repair_output":
            events.append({"event": "model_repair_output"})
    return events


def _empty_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "input": example["input"],
        "category": example["category"],
        "suite": example.get("suite", "normal"),
        "expected_route": example["expected_route"],
        "actual_route": "",
        "route_correct": False,
        **_tool_event_summary([]),
        "retry_count": 0,
        "replan_count": 0,
        "llm_calls": 0,
        "final_status": "",
        "stopped_reason": "",
        "final_answer": "",
        "task_success": False,
        "execution_contract_pass": False,
        "failure_stage": "",
        "execution_failure_stage": "",
        "failure_reasons": [],
        "task_failure_reasons": [],
        "execution_failure_reasons": [],
        "latency_ms": 0.0,
    }


def _observed_registry(registry: ToolRegistry) -> ToolRegistry:
    original_execute = registry.execute
    registry.observed_tool_events = []  # type: ignore[attr-defined]

    def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        registry.observed_tool_events.append({"tool": name, "event": "execution_attempt"})  # type: ignore[attr-defined]
        result = original_execute(name, arguments)
        if result.get("ok"):
            registry.observed_tool_events.append({"tool": name, "event": "execution_success"})  # type: ignore[attr-defined]
        else:
            registry.observed_tool_events.append(
                {"tool": name, "event": "execution_failure", "error": result.get("error")}
            )  # type: ignore[attr-defined]
        return result

    registry.execute = execute  # type: ignore[method-assign]
    return registry


def _registry_events(registry: ToolRegistry) -> list[dict[str, Any]]:
    return list(getattr(registry, "observed_tool_events", []))


def _tool_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return tool_event_summary(events)


def _runtime_confirmation(example: dict[str, Any]):
    def confirm(tool: str, arguments: dict[str, Any], risk: RiskLevel, reason: str) -> bool:
        return bool(example.get("confirmation", True))

    return confirm


def _plan_confirmation(example: dict[str, Any]):
    def confirm(plan: Plan, step, risk: RiskLevel, reason: str) -> bool:
        return bool(example.get("confirmation", True))

    return confirm


def _replan_callback(example: dict[str, Any], planner: StructuredPlanner):
    def replan(plan: Plan, reason: str) -> Plan:
        return planner.build_research_plan(example["input"], memory_context="")

    return replan


def _write_setup_files(workspace_dir: Path, setup: dict[str, Any]) -> None:
    for relative_path, content in (setup.get("files") or {}).items():
        path = _resolve_in(workspace_dir, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")


def _build_memory_service(example: dict[str, Any], workspace_dir: Path, settings) -> MemoryService | None:
    memories = (example.get("setup") or {}).get("memories") or []
    if not memories:
        return None
    store = MemoryStore(workspace_dir / "memory.sqlite3")
    service = MemoryService(store=store, summarizer=None, retrieval_limit=settings.memory_retrieval_limit)
    for item in memories:
        service.remember_explicit(str(item), source_run_id=example["id"])
    return service


@contextmanager
def _patched_workspace(workspace_dir: Path):
    original_file_resolver = file_tools_module.resolve_workspace_path
    original_executor_resolver = executor_module.resolve_workspace_path

    def resolve(relative_path: str, workspace_root: Path = workspace_dir) -> Path:
        return _resolve_in(workspace_root, relative_path)

    file_tools_module.resolve_workspace_path = resolve
    executor_module.resolve_workspace_path = resolve
    try:
        yield
    finally:
        file_tools_module.resolve_workspace_path = original_file_resolver
        executor_module.resolve_workspace_path = original_executor_resolver


def _resolve_in(root: Path, relative_path: str) -> Path:
    root_r = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root_r and root_r not in candidate.parents:
        raise ValueError("Path traversal outside workspace is not allowed")
    if candidate == root_r:
        raise ValueError("Path must refer to a file inside workspace")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live LLM end-to-end agent evaluation.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    client = CountingLLMClient(
        LocalLLMClient(
            base_url=settings.local_llm_base_url,
            api_key=settings.local_llm_api_key,
            model=settings.local_llm_model,
            timeout=settings.local_llm_timeout,
        )
    )
    examples = load_dataset(args.dataset)
    if args.limit is not None:
        examples = examples[: args.limit]

    run_dir = args.out_dir or RESULTS_ROOT / f"live_{time.strftime('%Y%m%d_%H%M%S')}"
    with tempfile.TemporaryDirectory(prefix="agent_live_e2e_") as tmp:
        root_dir = Path(tmp)
        records = []
        for example in examples:
            outcome = run_case(example, root_dir=root_dir, llm=client, settings=settings)
            records.append(outcome.record)
            status = "PASS" if outcome.record["task_success"] else "FAIL"
            print(
                f"{status} {outcome.record['id']} route={outcome.record['actual_route']} "
                f"stage={outcome.record['failure_stage'] or '-'} llm={outcome.record['llm_calls']} "
                f"latency_ms={outcome.record['latency_ms']}"
            )

    metadata = {
        "mode": "live_e2e",
        "dataset_version": DATASET_VERSION,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "dataset_fingerprint": dataset_fingerprint(args.dataset),
        "llm_model": settings.local_llm_model,
        "llm_provider": settings.local_llm_base_url,
        "llm_base_url": settings.local_llm_base_url,
        "router_configuration": "current RequestRouter; router tuning frozen",
        "key_config": (
            f"max_tool_calls={settings.max_tool_calls}; planner_max_steps={settings.planner_max_steps}; "
            f"planner_max_replans={settings.planner_max_replans}; tool_max_retries={settings.tool_max_retries}; "
            f"confirm_write_actions={settings.confirm_write_actions}; tavily_configured={bool(settings.tavily_api_key)}"
        ),
    }
    metrics = write_outputs(records, run_dir, metadata=metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
