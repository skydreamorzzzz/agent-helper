from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from src.agent.protocol import FinalAnswer, extract_first_json_object, parse_model_output
from src.logging_utils import log_event
from src.planner.models import ExecutionResult, Plan, PlanStatus, PlanStep, ReplanRecord, RiskLevel, StepStatus
from src.planner.prompts import (
    build_argument_resolution_prompt,
    build_cited_report_prompt,
    build_final_answer_prompt,
)
from src.planner.repository import PlanRepository
from src.planner.validator import PlanValidator
from src.tools.file_tools import resolve_workspace_path
from src.tools.policy import PolicyAction, ToolExecutionPolicy
from src.tools.registry import ToolRegistry


class ExecutorLLM(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


ConfirmationCallback = Callable[[Plan, PlanStep, RiskLevel, str], bool]
ReplanCallback = Callable[[Plan, str], Plan]


class PlanExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        repository: PlanRepository,
        llm_client: ExecutorLLM | None = None,
        validator: PlanValidator | None = None,
        max_retries: int = 1,
        max_replans: int = 2,
        confirm_write_actions: bool = True,
        confirmation_callback: ConfirmationCallback | None = None,
        replan_callback: ReplanCallback | None = None,
        logger: logging.Logger | None = None,
        tool_policy: ToolExecutionPolicy | None = None,
    ) -> None:
        self.registry = registry
        self.repository = repository
        self.llm_client = llm_client
        self.validator = validator or PlanValidator(registry)
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.confirm_write_actions = confirm_write_actions
        self.confirmation_callback = confirmation_callback
        self.replan_callback = replan_callback
        self.logger = logger
        self.tool_policy = tool_policy or ToolExecutionPolicy()

    def execute(self, plan: Plan) -> ExecutionResult:
        validation = self.validator.validate(plan)
        self._log("plan_validation", ok=validation.ok, errors=validation.errors)
        if not validation.ok:
            plan.status = PlanStatus.FAILED
            self.repository.save(plan)
            return ExecutionResult(plan=plan, final_answer="计划校验失败：" + "; ".join(validation.errors), stopped_reason="failed")

        plan.status = PlanStatus.RUNNING
        self._mark_ready_steps(plan)
        self.repository.save(plan)

        while plan.status == PlanStatus.RUNNING:
            step = self._next_ready_step(plan)
            if step is None:
                if all(step.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for step in plan.steps):
                    plan.status = PlanStatus.COMPLETED
                    self.repository.save(plan)
                    return ExecutionResult(plan=plan, final_answer=self._final_answer(plan), stopped_reason="completed")
                plan.status = PlanStatus.FAILED
                self.repository.save(plan)
                return ExecutionResult(plan=plan, final_answer="计划没有可执行的下一步。", stopped_reason="failed")

            plan.current_step_id = step.id
            step.status = StepStatus.RUNNING
            self.repository.save(plan)
            self._log("plan_step_started", plan_id=plan.plan_id, step_id=step.id, tool=step.tool_name, arguments=step.arguments)

            try:
                tool = self.registry.get(step.tool_name)
                resolved_arguments = self._resolve_arguments_if_needed(plan, step)
                tool.argument_schema.model_validate(resolved_arguments)
                decision = self.tool_policy.evaluate(
                    tool=tool,
                    arguments=resolved_arguments,
                    confirm_write_actions=self.confirm_write_actions,
                )
                self._log(
                    "tool_policy_decision",
                    plan_id=plan.plan_id,
                    step_id=step.id,
                    tool=step.tool_name,
                    risk_level=decision.risk_level,
                    decision=decision.action,
                    reason=decision.reason,
                )
                if decision.action == PolicyAction.DENY:
                    raise RuntimeError(f"Tool execution denied: {decision.reason}")
                if decision.action == PolicyAction.CONFIRM:
                    approved = self._confirm(plan, step, decision.risk_level, decision.reason)
                    self._log(
                        "tool_confirmation",
                        plan_id=plan.plan_id,
                        step_id=step.id,
                        tool=step.tool_name,
                        risk_level=decision.risk_level,
                        approved=approved,
                        reason=decision.reason,
                    )
                    if not approved:
                        step.status = StepStatus.PENDING
                        plan.status = PlanStatus.PAUSED
                        self.repository.save(plan)
                        self._log("plan_confirmation_required", plan_id=plan.plan_id, step_id=step.id, reason=decision.reason)
                        return ExecutionResult(
                            plan=plan,
                            final_answer=f"步骤 {step.id} 需要确认：{decision.reason}",
                            stopped_reason="confirmation_required",
                        )
                result = self._execute_tool(plan, step.tool_name, resolved_arguments)
                step.arguments = resolved_arguments
                if result.get("ok"):
                    step.actual_output = result["result"]
                    step.status = StepStatus.COMPLETED
                    step.error = None
                    self._log("plan_step_completed", plan_id=plan.plan_id, step_id=step.id, output=result)
                else:
                    raise RuntimeError(str(result.get("error", "tool failed")))
            except Exception as exc:
                step.retry_count += 1
                step.error = f"{type(exc).__name__}: {exc}"
                self._log("plan_step_failed", plan_id=plan.plan_id, step_id=step.id, error=step.error, retry_count=step.retry_count)
                if step.retry_count <= self.max_retries:
                    step.status = StepStatus.READY
                else:
                    step.status = StepStatus.FAILED
                    replanned = self._try_replan(plan, step.error)
                    if replanned is None:
                        plan.status = PlanStatus.FAILED
                        self.repository.save(plan)
                        reason_code = "replan_limit_reached" if plan.replan_count >= self.max_replans else "failed"
                        return ExecutionResult(plan=plan, final_answer=f"计划执行失败：{step.error}", stopped_reason=reason_code)
                    plan = replanned

            self._mark_ready_steps(plan)
            self.repository.save(plan)

        return ExecutionResult(plan=plan, final_answer=f"计划已停止：{plan.status}", stopped_reason="failed")

    def _mark_ready_steps(self, plan: Plan) -> None:
        completed = {step.id for step in plan.steps if step.status == StepStatus.COMPLETED}
        for step in plan.steps:
            if step.status == StepStatus.PENDING and all(dependency in completed for dependency in step.depends_on):
                step.status = StepStatus.READY

    def _next_ready_step(self, plan: Plan) -> PlanStep | None:
        for step in plan.steps:
            if step.status == StepStatus.READY:
                return step
        return None

    def _resolve_arguments_if_needed(self, plan: Plan, step: PlanStep) -> dict[str, Any]:
        if not self._contains_placeholder(step.arguments):
            return step.arguments
        if self.llm_client is None:
            raise ValueError(f"Step {step.id} has unresolved placeholders")
        tool = self.registry.get(step.tool_name)
        raw = self.llm_client.chat(
            [
                {
                    "role": "user",
                    "content": build_argument_resolution_prompt(
                        step_id=step.id,
                        tool_name=step.tool_name,
                        arguments=step.arguments,
                        observations=self._observations(plan),
                        tool_schema=tool.argument_schema.model_json_schema(),
                    ),
                }
            ]
        )
        json_text = extract_first_json_object(raw)
        if json_text is None:
            raise ValueError("Argument resolver returned no JSON")
        return json.loads(json_text)

    def _contains_placeholder(self, value: Any) -> bool:
        if isinstance(value, str):
            return "${" in value
        if isinstance(value, dict):
            return any(self._contains_placeholder(item) for item in value.values())
        if isinstance(value, list):
            return any(self._contains_placeholder(item) for item in value)
        return False

    def _execute_tool(self, plan: Plan, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "transform_text":
            return self._execute_transform_text(arguments)
        if tool_name == "write_cited_report":
            return self._execute_cited_report(plan, arguments)
        return self.registry.execute(tool_name, arguments)

    def _execute_transform_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.llm_client is None:
            return {"ok": False, "error": "transform_text requires llm_client"}
        prompt = (
            "Transform the input text according to the instruction. Return only the transformed text.\n"
            f"Instruction:\n{arguments['instruction']}\n\n"
            f"Input text:\n{arguments['input_text']}"
        )
        try:
            output = self.llm_client.chat([{"role": "user", "content": prompt}])
            return {"ok": True, "result": output.strip()}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _execute_cited_report(self, plan: Plan, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.llm_client is None:
            return {"ok": False, "error": "write_cited_report requires llm_client"}
        collected = self._research_materials(plan)
        if not collected:
            return {"ok": False, "error": "所有搜索均未返回结果，无法生成报告。"}
        try:
            report_file = str(arguments["report_file"])
            path = resolve_workspace_path(report_file)
            prompt = build_cited_report_prompt(
                topic=str(arguments["topic"]),
                sub_topics=[str(item) for item in arguments.get("sub_topics", [])],
                materials=json.dumps(collected, ensure_ascii=False, indent=2),
                report_file=report_file,
            )
            output = self.llm_client.chat([{"role": "user", "content": prompt}]).strip()
            if not output:
                return {"ok": False, "error": "报告生成为空。"}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(output, encoding="utf-8")
            source_count = sum(len(entry["results"]) for entry in collected)
            return {"ok": True, "result": f"报告已保存到 {report_file}（引用 {source_count} 个来源）"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _research_materials(self, plan: Plan) -> list[dict[str, Any]]:
        materials: list[dict[str, Any]] = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.tool_name == "search_web" and step.actual_output:
                try:
                    results = (
                        json.loads(step.actual_output)
                        if isinstance(step.actual_output, str)
                        else step.actual_output
                    )
                except (json.JSONDecodeError, TypeError):
                    results = []
                materials.append({"sub_topic": step.description.removeprefix("搜索："), "results": results})
        return materials

    def _confirm(self, plan: Plan, step: PlanStep, risk: RiskLevel, reason: str) -> bool:
        if self.confirmation_callback is None:
            return False
        return self.confirmation_callback(plan, step, risk, reason)

    def _try_replan(self, plan: Plan, reason: str) -> Plan | None:
        if plan.replan_count >= self.max_replans or self.replan_callback is None:
            self._log("plan_replan_stopped", plan_id=plan.plan_id, reason=reason, replan_count=plan.replan_count)
            return None
        original = plan.model_dump_json()
        new_plan = self.replan_callback(plan, reason)
        new_plan.plan_id = plan.plan_id
        new_plan.created_at = plan.created_at
        new_plan.replan_count = plan.replan_count + 1
        new_plan.replan_history = [
            *plan.replan_history,
            ReplanRecord(reason=reason, original_plan_json=original),
        ]
        completed_by_id = {step.id: step for step in plan.steps if step.status == StepStatus.COMPLETED}
        for step in new_plan.steps:
            if step.id in completed_by_id:
                completed = completed_by_id[step.id]
                step.status = StepStatus.COMPLETED
                step.actual_output = completed.actual_output
                step.error = None
        new_plan.status = PlanStatus.RUNNING
        self._log("plan_replanned", plan_id=plan.plan_id, reason=reason, replan_count=new_plan.replan_count)
        self.repository.save(new_plan)
        return new_plan

    def _final_answer(self, plan: Plan) -> str:
        observations = self._observations(plan)
        if self.llm_client is not None:
            try:
                raw = self.llm_client.chat(
                    [{"role": "user", "content": build_final_answer_prompt(plan.goal, observations, plan.final_output_requirement)}]
                )
                parsed = parse_model_output(raw)
                if isinstance(parsed, FinalAnswer):
                    return parsed.content
            except Exception:
                pass
        return f"计划已完成。执行了 {len([step for step in plan.steps if step.status == StepStatus.COMPLETED])} 个步骤。"

    def _observations(self, plan: Plan) -> dict[str, object]:
        return {
            step.id: {
                "description": step.description,
                "tool_name": step.tool_name,
                "arguments": step.arguments,
                "actual_output": step.actual_output,
                "error": step.error,
            }
            for step in plan.steps
            if step.status in (StepStatus.COMPLETED, StepStatus.FAILED)
        }

    def _log(self, event: str, **data: Any) -> None:
        if self.logger is not None:
            log_event(self.logger, event, **data)
