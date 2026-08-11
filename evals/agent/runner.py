from __future__ import annotations

import argparse
import ast
import json
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agent.runtime import AgentRuntime
from src.cli import build_registry, build_research_registry
from src.memory.service import MemoryService
from src.memory.store import MemoryStore
from src.planner.executor import PlanExecutor
from src.planner.models import Plan, RiskLevel, Route, StepStatus
from src.planner.planner import StructuredPlanner
from src.planner.repository import PlanRepository
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.base import Tool, ToolExecutionError
from src.tools.registry import ToolRegistry

from evals.agent.evaluators import evaluate_task
from evals.agent.report import git_commit, write_outputs

import src.planner.executor as executor_module
import src.tools.file_tools as file_tools_module


DATASET_PATH = Path(__file__).with_name("dataset.jsonl")
RESULTS_ROOT = Path(__file__).with_name("results")
DATASET_VERSION = "agent-e2e-v1"


class FakeSearchArguments(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)


class FakeSearchWebTool(Tool):
    name = "search_web"
    description = "Deterministic fake web search for E2E evaluation."
    argument_schema = FakeSearchArguments
    risk_level = RiskLevel.READ_ONLY

    def __init__(self, example: dict[str, Any]) -> None:
        self.example = example
        self.calls = 0

    def execute(self, arguments: FakeSearchArguments) -> str:
        self.calls += 1
        fake = self.example.get("fake") or {}
        if fake.get("search_empty"):
            raise ToolExecutionError(f"搜索「{arguments.query}」没有返回任何结果。")
        if fake.get("search_fail_once") and self.calls == 1:
            raise ToolExecutionError("deterministic transient search failure")
        query = arguments.query
        results = [
            {
                "title": f"{query} source 1",
                "url": f"https://example.test/{self.calls}/source-1",
                "content": f"{query} 的确定性搜索摘要，包含 Tavily、SerpAPI、Agent Memory、Python、OpenAI 或 pricing 等关键词。",
            },
            {
                "title": f"{query} source 2",
                "url": f"https://example.test/{self.calls}/source-2",
                "content": f"{query} 的第二条结果，用于生成带来源的报告或回答。",
            },
        ]
        return json.dumps(results[: arguments.max_results], ensure_ascii=False)


@dataclass
class CountingFakeLLM:
    example: dict[str, Any]
    workspace_dir: Path
    count: int = 0
    runtime_turns: int = 0
    last_transform_output: str = ""
    emitted_tool_calls: list[str] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.count += 1
        content = messages[-1]["content"]
        if content.startswith("Classify the user request"):
            return self._route_response(content)
        if content.startswith("Create a structured executable plan"):
            return self._plan_response(content)
        if content.startswith("You are planning an in-depth web research task"):
            return self._research_plan_response(content)
        if content.startswith("Resolve tool arguments"):
            return self._argument_response(content)
        if content.startswith("Transform the input text according to the instruction"):
            self.last_transform_output = self._transform_response(content)
            return self.last_transform_output
        if content.startswith("You are writing a deep-research report"):
            return self._cited_report_response(content)
        if content.startswith("Generate the final answer for a completed plan"):
            return self._plan_final_answer(content)
        if content.startswith("Your previous response was invalid"):
            return json.dumps({"type": "final_answer", "content": "已完成无效 JSON 恢复测试。"}, ensure_ascii=False)
        return self._runtime_response(messages)

    def _route_response(self, content: str) -> str:
        return json.dumps(
            {
                "route": self.example["expected_route"],
                "reason": "deterministic fake router response",
                "missing_information": ["请提供缺失的关键参数，例如文件名、输入内容或保存位置。"]
                if self.example["expected_route"] == "clarification"
                else [],
            },
            ensure_ascii=False,
        )

    def _plan_response(self, content: str) -> str:
        user_input = _extract_after(content, "User request:")
        fake = self.example.get("fake") or {}
        if fake.get("planner_bad_placeholder"):
            return json.dumps(
                {
                    "goal": user_input,
                    "steps": [
                        {
                            "id": "step_1",
                            "description": "读取输入文件",
                            "tool_name": "read_text_file",
                            "arguments": {"path": "notes.txt"},
                            "depends_on": [],
                            "expected_output": "原始内容",
                        },
                        {
                            "id": "step_2",
                            "description": "写入错误占位符结果",
                            "tool_name": "write_text_file",
                            "arguments": {"path": "bad_summary.md", "content": "${step_1.result.top_task_1}"},
                            "depends_on": ["step_1"],
                            "expected_output": "错误摘要",
                        },
                    ],
                    "assumptions": [],
                    "unresolved_questions": [],
                    "final_output_requirement": "保存摘要文件。",
                },
                ensure_ascii=False,
            )
        if "计算" in user_input and "保存" in user_input:
            expression = _extract_expression(user_input) or "19 * 23"
            output_file = _extract_filename(user_input) or "calc.md"
            steps = [
                {
                    "id": "step_1",
                    "description": "计算表达式",
                    "tool_name": "calculator",
                    "arguments": {"expression": expression},
                    "depends_on": [],
                    "expected_output": "计算结果",
                },
                {
                    "id": "step_2",
                    "description": "保存计算结果",
                    "tool_name": "write_text_file",
                    "arguments": {"path": output_file, "content": "${step_1.result}"},
                    "depends_on": ["step_1"],
                    "expected_output": "写入文件",
                },
            ]
        else:
            input_file = _extract_filename(user_input, extensions=("txt",)) or "notes.txt"
            output_file = _extract_filename(user_input, extensions=("md", "txt"), prefer_last=True) or "summary.md"
            instruction = "总结三点"
            if "最大值" in user_input:
                instruction = "找出最大值"
            elif "更短" in user_input or "整理" in user_input:
                instruction = "整理成更短版本"
            steps = [
                {
                    "id": "step_1",
                    "description": "读取输入文件",
                    "tool_name": "read_text_file",
                    "arguments": {"path": input_file},
                    "depends_on": [],
                    "expected_output": "原始文本",
                },
                {
                    "id": "step_2",
                    "description": "整理文本",
                    "tool_name": "transform_text",
                    "arguments": {"input_text": "${step_1.result}", "instruction": instruction},
                    "depends_on": ["step_1"],
                    "expected_output": "整理后的文本",
                },
                {
                    "id": "step_3",
                    "description": "保存结果",
                    "tool_name": "write_text_file",
                    "arguments": {"path": output_file, "content": "${step_2.result}"},
                    "depends_on": ["step_2"],
                    "expected_output": "写入文件",
                },
            ]
        return json.dumps(
            {
                "goal": user_input,
                "steps": steps,
                "assumptions": [],
                "unresolved_questions": [],
                "final_output_requirement": "完成任务并简短说明结果。",
            },
            ensure_ascii=False,
        )

    def _research_plan_response(self, content: str) -> str:
        user_input = _extract_after(content, "User request:")
        if "Tavily" in user_input or "SerpAPI" in user_input:
            report_file = "reports/search_api_compare.md"
            topic = "Tavily 和 SerpAPI 搜索 API 差异"
            sub_topics = [
                {"question": "Tavily 搜索 API", "search_queries": ["Tavily search API pricing features"]},
                {"question": "SerpAPI 搜索 API", "search_queries": ["SerpAPI pricing features"]},
            ]
        elif "失败搜索" in user_input:
            report_file = "reports/retry_research.md"
            topic = "失败搜索案例"
            sub_topics = [{"question": "失败搜索恢复", "search_queries": ["transient search failure retry"]}]
        else:
            report_file = "reports/agent_memory.md"
            topic = "Agent Memory 方法"
            sub_topics = [
                {"question": "Agent Memory 类型", "search_queries": ["Agent Memory methods"]},
                {"question": "Agent Memory 优缺点", "search_queries": ["Agent Memory pros and cons"]},
            ]
        return json.dumps(
            {"topic": topic, "sub_topics": sub_topics, "report_file": report_file, "assumptions": []},
            ensure_ascii=False,
        )

    def _argument_response(self, content: str) -> str:
        tool = _extract_line_value(content, "Tool:")
        observations = _extract_observations(content)
        if tool == "transform_text":
            previous = _first_actual_output(observations)
            instruction = "总结三点"
            if "最大值" in content:
                instruction = "找出最大值"
            elif "更短" in content or "整理" in content:
                instruction = "整理成更短版本"
            return json.dumps({"input_text": previous, "instruction": instruction}, ensure_ascii=False)
        if tool == "write_text_file":
            current = _extract_current_arguments(content)
            path = str(current.get("path") or "output.md")
            previous = _last_actual_output(observations) or self.last_transform_output
            return json.dumps({"path": path, "content": previous, "overwrite": bool(current.get("overwrite", False))}, ensure_ascii=False)
        return "{}"

    def _transform_response(self, content: str) -> str:
        instruction = _extract_between(content, "Instruction:\n", "\n\nInput text:") or ""
        input_text = _extract_after(content, "Input text:\n")
        if "最大值" in instruction:
            numbers = [int(item) for item in re.findall(r"-?\d+", input_text)]
            return str(max(numbers)) if numbers else "未找到数字"
        if "更短" in instruction:
            return input_text.strip()[:80]
        lines = [line.strip(" -：:") for line in input_text.splitlines() if line.strip()]
        return "\n".join(f"- {line}" for line in lines[:3])

    def _cited_report_response(self, content: str) -> str:
        topic = _extract_line_value(content, "Topic:") or "Agent Memory"
        urls = re.findall(r"https?://[^\s\"')\]]+", content)
        unique_urls = []
        for url in urls:
            if url not in unique_urls:
                unique_urls.append(url)
        cited = unique_urls[:2] or ["https://example.test/source"]
        return (
            f"# {topic}\n\n"
            f"本报告总结 {topic} 的关键结论，并只引用搜索材料中的来源。\n\n"
            f"## 主要发现\n\n"
            f"- {topic} 与 Agent Memory、Tavily、SerpAPI 或 Python 版本等信息相关。[来源]({cited[0]})\n"
            f"- 对比时需要关注功能、成本、可靠性和集成复杂度。[来源]({cited[-1]})\n\n"
            "## 结论\n\n"
            "建议根据任务复杂度和可观测性要求选择方案。\n\n"
            "## 参考来源\n\n"
            + "\n".join(f"{index + 1}. {url}" for index, url in enumerate(cited))
        )

    def _plan_final_answer(self, content: str) -> str:
        return json.dumps({"type": "final_answer", "content": "计划已完成，产物已保存。"}, ensure_ascii=False)

    def _runtime_response(self, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        fake = self.example.get("fake") or {}
        if fake.get("invalid_first_runtime_output") and self.runtime_turns == 0:
            self.runtime_turns += 1
            return "this is not json"
        if content.startswith("Tool result JSON:"):
            payload = _extract_tool_result(content)
            result = payload.get("result") or payload.get("error") or ""
            return json.dumps({"type": "final_answer", "content": str(result)}, ensure_ascii=False)
        self.runtime_turns += 1
        user_input = _extract_after(content, "Current User Request\n====================\n") or self.example["input"]
        if self.example["expected_route"] == "web_lookup":
            self.emitted_tool_calls.append("search_web")
            return json.dumps(
                {"type": "tool_call", "tool": "search_web", "arguments": {"query": user_input, "max_results": 3}},
                ensure_ascii=False,
            )
        if "计算" in user_input or re.search(r"\bcalculate\b", user_input, re.I):
            self.emitted_tool_calls.append("calculator")
            return json.dumps(
                {"type": "tool_call", "tool": "calculator", "arguments": {"expression": _extract_expression(user_input) or user_input}},
                ensure_ascii=False,
            )
        if "读取" in user_input or re.search(r"\bread\b", user_input, re.I):
            self.emitted_tool_calls.append("read_text_file")
            return json.dumps(
                {"type": "tool_call", "tool": "read_text_file", "arguments": {"path": _extract_filename(user_input) or "missing.txt"}},
                ensure_ascii=False,
            )
        if "保存" in user_input or "写入" in user_input or re.search(r"\bwrite\b", user_input, re.I):
            path = _extract_filename(user_input) or "output.txt"
            content_to_write = _extract_write_content(user_input)
            overwrite = "覆盖" in user_input or "overwrite" in user_input.lower()
            self.emitted_tool_calls.append("write_text_file")
            return json.dumps(
                {
                    "type": "tool_call",
                    "tool": "write_text_file",
                    "arguments": {"path": path, "content": content_to_write, "overwrite": overwrite},
                },
                ensure_ascii=False,
            )
        answer = "我是本地个人助手，可以回答问题、调用工具、处理本地文件，并结合记忆上下文。"
        if "残差连接" in user_input:
            answer = "残差连接让输入绕过若干层后相加，有助于梯度传播和训练深层 Transformer。"
        if "agent-helper" in _memory_text(messages) or "项目叫什么" in user_input:
            answer = "你正在开发的项目是 agent-helper。"
        return json.dumps({"type": "final_answer", "content": answer}, ensure_ascii=False)


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


def run_case(example: dict[str, Any], *, root_dir: Path) -> CaseOutcome:
    workspace_dir = root_dir / example["id"] / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    _write_setup_files(workspace_dir, example.get("setup") or {})
    llm = CountingFakeLLM(example=example, workspace_dir=workspace_dir)
    memory_service = _build_memory_service(example, workspace_dir)
    record = _empty_record(example)
    start = time.perf_counter()

    with _patched_workspace(workspace_dir):
        try:
            record.update(_execute_pipeline(example, llm, workspace_dir, memory_service))
        except Exception as exc:
            record["final_status"] = "runner_error"
            record["stopped_reason"] = f"runner_error:{type(exc).__name__}: {exc}"
    record["llm_calls"] = llm.count
    record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    evaluation = evaluate_task(example, record, workspace_dir)
    record["task_success"] = evaluation.task_success
    record["failure_stage"] = evaluation.failure_stage
    record["failure_reasons"] = evaluation.failure_reasons
    return CaseOutcome(record=record, workspace_dir=workspace_dir)


def _execute_pipeline(
    example: dict[str, Any],
    llm: CountingFakeLLM,
    workspace_dir: Path,
    memory_service: MemoryService | None,
) -> dict[str, Any]:
    core_registry = _observed_registry(build_registry())
    lookup_registry = _build_lookup_registry(example)
    research_registry = _build_research_registry(example)
    router = RequestRouter(llm)
    memory_context = memory_service.retrieve_context(example["input"]).block if memory_service else ""
    decision = router.route(example["input"], memory_context=memory_context)
    record = {"actual_route": str(decision.route)}
    if decision.route == Route.CLARIFICATION:
        record.update(
            {
                "final_status": "clarification",
                "stopped_reason": "clarification",
                "final_answer": " ".join(decision.missing_information),
                "tool_calls": [],
                "tool_failures": [],
            }
        )
        return record
    if decision.route == Route.WEB_LOOKUP:
        runtime_result = AgentRuntime(
            llm_client=llm,
            tool_registry=lookup_registry,
            max_tool_calls=5,
            memory_service=memory_service,
            confirm_write_actions=True,
            confirmation_callback=_runtime_confirmation(example),
        ).run(
            example["input"],
            required_tool="search_web",
            execution_policy="This request was routed as web_lookup. You must call search_web before final_answer.",
        )
        record.update(_runtime_record(runtime_result.content, runtime_result.stopped_reason, lookup_registry, llm))
        return record
    if decision.route in (Route.PLANNED_TASK, Route.DEEP_RESEARCH):
        registry = research_registry if decision.route == Route.DEEP_RESEARCH else core_registry
        planner = StructuredPlanner(llm, registry, max_steps=8)
        repo = PlanRepository(workspace_dir / "plans.sqlite3")
        validator = PlanValidator(registry, max_steps=8)
        try:
            plan = (
                planner.build_research_plan(example["input"], memory_context=memory_context)
                if decision.route == Route.DEEP_RESEARCH
                else planner.create_plan(example["input"], memory_context=memory_context)
            )
        except Exception as exc:
            record.update(
                {
                    "final_status": "planning_failed",
                    "stopped_reason": f"planning_failed:{type(exc).__name__}: {exc}",
                    "tool_calls": [],
                    "tool_failures": [],
                }
            )
            return record
        result = PlanExecutor(
            registry=registry,
            repository=repo,
            llm_client=llm,
            validator=validator,
            max_retries=1,
            max_replans=2,
            confirm_write_actions=True,
            confirmation_callback=_plan_confirmation(example),
            replan_callback=_replan_callback(example, planner) if decision.route == Route.DEEP_RESEARCH else None,
        ).execute(plan)
        record.update(_plan_record(result.plan, result.final_answer, result.stopped_reason))
        return record
    runtime_result = AgentRuntime(
        llm_client=llm,
        tool_registry=core_registry,
        max_tool_calls=5,
        memory_service=memory_service,
        confirm_write_actions=True,
        confirmation_callback=_runtime_confirmation(example),
    ).run(example["input"])
    record.update(_runtime_record(runtime_result.content, runtime_result.stopped_reason, core_registry, llm))
    return record


def _runtime_record(final_answer: str, stopped_reason: str, registry: ToolRegistry, llm: CountingFakeLLM) -> dict[str, Any]:
    executed = list(getattr(registry, "observed_tool_calls", []))
    tool_calls = list(dict.fromkeys([*llm.emitted_tool_calls, *executed]))
    return {
        "final_status": stopped_reason,
        "stopped_reason": stopped_reason,
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "tool_failures": getattr(registry, "observed_tool_failures", []),
        "retry_count": 0,
        "replan_count": 0,
    }


def _plan_record(plan: Plan, final_answer: str, stopped_reason: str) -> dict[str, Any]:
    return {
        "final_status": stopped_reason,
        "stopped_reason": stopped_reason,
        "final_answer": final_answer,
        "tool_calls": [step.tool_name for step in plan.steps if step.status in (StepStatus.COMPLETED, StepStatus.FAILED)],
        "tool_failures": [
            {"step_id": step.id, "tool": step.tool_name, "error": step.error}
            for step in plan.steps
            if step.status == StepStatus.FAILED
        ],
        "retry_count": sum(step.retry_count for step in plan.steps),
        "replan_count": plan.replan_count,
    }


def _empty_record(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": example["id"],
        "input": example["input"],
        "category": example["category"],
        "expected_route": example["expected_route"],
        "actual_route": "",
        "tool_calls": [],
        "tool_failures": [],
        "retry_count": 0,
        "replan_count": 0,
        "llm_calls": 0,
        "final_status": "",
        "stopped_reason": "",
        "final_answer": "",
        "task_success": False,
        "failure_stage": "",
        "failure_reasons": [],
        "latency_ms": 0.0,
    }


def _build_lookup_registry(example: dict[str, Any]) -> ToolRegistry:
    registry = build_registry()
    registry.register(FakeSearchWebTool(example))
    return _observed_registry(registry)


def _build_research_registry(example: dict[str, Any]) -> ToolRegistry:
    registry = build_research_registry(api_key="")
    registry._tools["search_web"] = FakeSearchWebTool(example)
    return _observed_registry(registry)


def _observed_registry(registry: ToolRegistry) -> ToolRegistry:
    original_execute = registry.execute
    registry.observed_tool_calls = []  # type: ignore[attr-defined]
    registry.observed_tool_failures = []  # type: ignore[attr-defined]

    def execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        registry.observed_tool_calls.append(name)  # type: ignore[attr-defined]
        result = original_execute(name, arguments)
        if not result.get("ok"):
            registry.observed_tool_failures.append({"tool": name, "error": result.get("error")})  # type: ignore[attr-defined]
        return result

    registry.execute = execute  # type: ignore[method-assign]
    return registry


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


def _build_memory_service(example: dict[str, Any], workspace_dir: Path) -> MemoryService | None:
    memories = (example.get("setup") or {}).get("memories") or []
    if not memories:
        return None
    store = MemoryStore(workspace_dir / "memory.sqlite3")
    service = MemoryService(store=store, summarizer=None)
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


def _extract_after(text: str, marker: str) -> str:
    index = text.rfind(marker)
    if index < 0:
        return ""
    return text[index + len(marker) :].strip()


def _extract_between(text: str, start: str, end: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        return ""
    start_index += len(start)
    end_index = text.find(end, start_index)
    if end_index < 0:
        return text[start_index:].strip()
    return text[start_index:end_index].strip()


def _extract_line_value(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.startswith(label):
            return line[len(label) :].strip()
    return ""


def _extract_filename(text: str, extensions: tuple[str, ...] = ("txt", "md"), prefer_last: bool = False) -> str:
    pattern = r"[\w\-]+\.(?:" + "|".join(re.escape(ext) for ext in extensions) + r")"
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    return matches[-1] if prefer_last else matches[0]


def _extract_expression(text: str) -> str:
    blocked = re.search(r"__import__\([^)]*\)\.system\([^)]*\)", text)
    if blocked:
        return blocked.group(0)
    match = re.search(r"\d+(?:\.\d+)?\s*(?:\+|\-|\*|/|//|%)\s*\d+(?:\.\d+)?", text)
    return match.group(0) if match else ""


def _extract_write_content(text: str) -> str:
    if "：" in text:
        return text.rsplit("：", 1)[-1].strip()
    if ":" in text:
        return text.rsplit(":", 1)[-1].strip()
    if "new content" in text:
        return "new content"
    return "deterministic content"


def _extract_tool_result(text: str) -> dict[str, Any]:
    payload = _extract_between(text, "Tool result JSON:\n", "\nReturn the next agent JSON object.")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _extract_current_arguments(text: str) -> dict[str, Any]:
    match = re.search(r"Current arguments: (\{.*?\})\nPrior observations:", text, re.S)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return {}


def _extract_observations(text: str) -> dict[str, Any]:
    match = re.search(r"Prior observations: (\{.*\})\nTool argument schema:", text, re.S)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group(1))
    except Exception:
        return {}


def _first_actual_output(observations: dict[str, Any]) -> str:
    for item in observations.values():
        if isinstance(item, dict) and item.get("actual_output") is not None:
            return str(item["actual_output"])
    return ""


def _last_actual_output(observations: dict[str, Any]) -> str:
    for item in reversed(list(observations.values())):
        if isinstance(item, dict) and item.get("actual_output") is not None:
            return str(item["actual_output"])
    return ""


def _memory_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message.get("content", "") for message in messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic end-to-end agent evaluation.")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="dataset JSONL path")
    parser.add_argument("--out", default=str(RESULTS_ROOT), help="output root")
    parser.add_argument("--limit", type=int, default=0, help="optional max examples")
    args = parser.parse_args()

    examples = load_dataset(Path(args.dataset))
    if args.limit:
        examples = examples[: args.limit]
    out_dir = Path(args.out) / time.strftime("run_%Y%m%d_%H%M%S")
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="agent_e2e_") as tmp:
        root_dir = Path(tmp)
        for index, example in enumerate(examples, 1):
            print(f"[{index}/{len(examples)}] {example['id']} {example['category']}", flush=True)
            outcome = run_case(example, root_dir=root_dir)
            records.append(outcome.record)
            print(
                f"  route={outcome.record['actual_route']} success={outcome.record['task_success']} "
                f"stage={outcome.record['failure_stage'] or '-'}",
                flush=True,
            )
    metadata = {
        "mode": "deterministic_e2e",
        "dataset_version": DATASET_VERSION,
        "git_commit": git_commit(),
        "llm_model": "CountingFakeLLM",
        "router_configuration": "current RequestRouter; no Router threshold/prototype changes in this run",
    }
    metrics = write_outputs(records, out_dir, metadata=metadata)
    print(f"Done. Report: {out_dir / 'report.md'}")
    print(f"Overall task success: {metrics['overall_task_success_rate']:.1%}")


if __name__ == "__main__":
    main()
