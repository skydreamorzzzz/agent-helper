from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from src.agent.prompts import build_repair_prompt, build_system_prompt
from src.agent.protocol import FinalAnswer, ModelOutputParseError, ToolCall, parse_model_output
from src.config import LOGS_DIR
from src.logging_utils import log_event, setup_run_logger
from src.memory.service import MemoryService
from src.planner.models import RiskLevel
from src.tools.policy import PolicyAction, ToolExecutionPolicy
from src.tools.registry import ToolArgumentError, ToolRegistry, UnknownToolError


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class AgentResult:
    run_id: str
    content: str
    stopped_reason: str


RuntimeConfirmationCallback = Callable[[str, dict[str, Any], RiskLevel, str], bool]


class AgentRuntime:
    def __init__(
        self,
        *,
        llm_client: ChatClient,
        tool_registry: ToolRegistry,
        max_tool_calls: int = 5,
        memory_service: MemoryService | None = None,
        working_memory_max_messages: int = 12,
        working_memory_max_chars: int = 12000,
        summary_trigger_messages: int = 16,
        confirm_write_actions: bool = True,
        confirmation_callback: RuntimeConfirmationCallback | None = None,
        tool_policy: ToolExecutionPolicy | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_tool_calls = max_tool_calls
        self.memory_service = memory_service
        self.working_memory_max_messages = working_memory_max_messages
        self.working_memory_max_chars = working_memory_max_chars
        self.summary_trigger_messages = summary_trigger_messages
        self.confirm_write_actions = confirm_write_actions
        self.confirmation_callback = confirmation_callback
        self.tool_policy = tool_policy or ToolExecutionPolicy()
        self.messages: list[dict[str, str]] = []

    def run(
        self,
        user_input: str,
        *,
        required_tool: str | None = None,
        execution_policy: str = "",
    ) -> AgentResult:
        run_id = uuid.uuid4().hex
        logger = setup_run_logger(run_id, LOGS_DIR)
        log_event(
            logger,
            "run_started",
            run_id=run_id,
            user_input=user_input,
            required_tool=required_tool,
        )

        memory_context = None
        recent_messages = self.messages
        conversation_summary = ""
        if self.memory_service is not None:
            memory_context = self.memory_service.retrieve_context(user_input)
            recent_messages = self.memory_service.recent_messages
            conversation_summary = self.memory_service.conversation_summary
            log_event(
                logger,
                "memory_retrieval",
                memories=[
                    {
                        "id": item.memory.id,
                        "score": item.score,
                        "source_run_id": item.memory.source_run_id,
                        "category": item.memory.category,
                    }
                    for item in memory_context.retrieved
                ],
            )

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.tool_registry,
                    memory_context=memory_context.block if memory_context else "No memory service configured.",
                    conversation_summary=conversation_summary,
                    execution_policy=execution_policy,
                ),
            },
            *recent_messages,
            {
                "role": "user",
                "content": (
                    "Current User Request\n"
                    "====================\n"
                    f"{user_input}"
                ),
            },
        ]

        tool_calls = 0
        required_tool_satisfied = required_tool is None
        required_tool_reminders = 0
        while tool_calls <= self.max_tool_calls:
            try:
                raw_output = self.llm_client.chat(messages)
            except Exception as exc:
                content = f"模型调用失败，已安全停止：{type(exc).__name__}: {exc}"
                log_event(logger, "llm_call_failed", error=str(exc), error_type=type(exc).__name__)
                self._append_turn(user_input, content)
                return AgentResult(run_id, content, "llm_call_failed")
            log_event(logger, "model_output", raw=raw_output)

            try:
                parsed = parse_model_output(raw_output)
            except ModelOutputParseError as exc:
                log_event(logger, "model_parse_failed", raw=raw_output, error=str(exc))
                repaired = self._repair_output(messages, raw_output, str(exc), logger)
                if repaired is None:
                    content = "模型输出不是合法的 Agent JSON，且一次格式修复失败。已安全停止。"
                    log_event(logger, "run_failed", reason="invalid_json_repair_failed")
                    self._append_turn(user_input, content)
                    return AgentResult(run_id, content, "invalid_json_repair_failed")
                parsed = repaired
                raw_output = parsed.model_dump_json()

            if isinstance(parsed, FinalAnswer):
                if not required_tool_satisfied:
                    if required_tool_reminders >= 1:
                        content = f"必须先成功调用 {required_tool} 才能回答，但模型仍直接返回最终答案，已安全停止。"
                        log_event(logger, "required_tool_missing", required_tool=required_tool)
                        self._append_turn(user_input, content)
                        return AgentResult(run_id, content, "required_tool_missing")
                    required_tool_reminders += 1
                    log_event(
                        logger,
                        "final_answer_rejected",
                        required_tool=required_tool,
                        content=parsed.content,
                    )
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"This run requires a successful {required_tool} tool call before final_answer. "
                                f"Call {required_tool} now with arguments based on the current user request. "
                                "Return only the next agent JSON object."
                            ),
                        }
                    )
                    continue
                log_event(logger, "final_answer", content=parsed.content)
                self._append_turn(user_input, parsed.content)
                return AgentResult(run_id, parsed.content, "final_answer")

            if isinstance(parsed, ToolCall):
                if tool_calls >= self.max_tool_calls:
                    content = f"已达到最大工具调用轮数 {self.max_tool_calls}，已安全停止。"
                    log_event(logger, "max_tool_calls_reached", max_tool_calls=self.max_tool_calls)
                    self._append_turn(user_input, content)
                    return AgentResult(run_id, content, "max_tool_calls_reached")

                tool_calls += 1
                log_event(
                    logger,
                    "tool_call",
                    tool=parsed.tool,
                    arguments=parsed.arguments,
                    index=tool_calls,
                )
                tool_result, stopped_reason = self._execute_tool(parsed, logger)
                if stopped_reason is not None:
                    content = json.loads(tool_result).get("error", "工具执行被策略阻止，已安全停止。")
                    self._append_turn(user_input, content)
                    return AgentResult(run_id, content, stopped_reason)
                if parsed.tool == required_tool:
                    try:
                        required_tool_satisfied = bool(json.loads(tool_result).get("ok"))
                    except json.JSONDecodeError:
                        required_tool_satisfied = False
                messages.append({"role": "assistant", "content": raw_output})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool result JSON:\n"
                            f"{tool_result}\n"
                            "Return the next agent JSON object."
                        ),
                    }
                )

        content = f"已达到最大工具调用轮数 {self.max_tool_calls}，已安全停止。"
        log_event(logger, "max_tool_calls_reached", max_tool_calls=self.max_tool_calls)
        self._append_turn(user_input, content)
        return AgentResult(run_id, content, "max_tool_calls_reached")

    def _repair_output(
        self,
        messages: list[dict[str, str]],
        raw_output: str,
        error: str,
        logger: logging.Logger,
    ) -> ToolCall | FinalAnswer | None:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": raw_output},
            {"role": "user", "content": build_repair_prompt(raw_output, error)},
        ]
        try:
            repaired_raw = self.llm_client.chat(repair_messages)
        except Exception as exc:
            log_event(logger, "model_repair_call_failed", error=str(exc), error_type=type(exc).__name__)
            return None
        log_event(logger, "model_repair_output", raw=repaired_raw)
        try:
            return parse_model_output(repaired_raw)
        except ModelOutputParseError as exc:
            log_event(logger, "model_repair_failed", raw=repaired_raw, error=str(exc))
            return None

    def _execute_tool(self, tool_call: ToolCall, logger: logging.Logger) -> tuple[str, str | None]:
        try:
            tool = self.tool_registry.get(tool_call.tool)
            normalized_arguments = self.tool_registry.normalize_arguments(tool_call.tool, tool_call.arguments)
            decision = self.tool_policy.evaluate(
                tool=tool,
                arguments=normalized_arguments,
                confirm_write_actions=self.confirm_write_actions,
            )
            log_event(
                logger,
                "tool_policy_decision",
                tool=tool_call.tool,
                risk_level=decision.risk_level,
                decision=decision.action,
                reason=decision.reason,
            )
            if decision.action == PolicyAction.DENY:
                result = {"ok": False, "error": f"Tool execution denied: {decision.reason}"}
                log_event(logger, "tool_result", tool=tool_call.tool, result=result)
                return json.dumps(result, ensure_ascii=False), "tool_policy_denied"
            if decision.action == PolicyAction.CONFIRM:
                approved = False
                if self.confirmation_callback is not None:
                    approved = self.confirmation_callback(
                        tool_call.tool,
                        normalized_arguments,
                        decision.risk_level,
                        decision.reason,
                    )
                log_event(
                    logger,
                    "tool_confirmation",
                    tool=tool_call.tool,
                    risk_level=decision.risk_level,
                    approved=approved,
                    reason=decision.reason,
                )
                if not approved:
                    result = {"ok": False, "error": f"Tool execution requires confirmation: {decision.reason}"}
                    log_event(logger, "tool_result", tool=tool_call.tool, result=result)
                    return json.dumps(result, ensure_ascii=False), "tool_confirmation_rejected"
            result = self.tool_registry.execute(tool_call.tool, normalized_arguments)
        except (UnknownToolError, ToolArgumentError) as exc:
            result = {"ok": False, "error": str(exc)}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log_event(logger, "tool_result", tool=tool_call.tool, result=result)
        return json.dumps(result, ensure_ascii=False), None

    def _append_turn(self, user_input: str, assistant_output: str) -> None:
        if self.memory_service is not None:
            self.memory_service.append_recent_turn(
                user_input,
                assistant_output,
                max_messages=self.working_memory_max_messages,
                max_chars=self.working_memory_max_chars,
                summary_trigger_messages=self.summary_trigger_messages,
            )
            return
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": assistant_output})
