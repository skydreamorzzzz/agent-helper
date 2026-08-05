from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Protocol

from src.agent.prompts import build_repair_prompt, build_system_prompt
from src.agent.protocol import FinalAnswer, ModelOutputParseError, ToolCall, parse_model_output
from src.config import LOGS_DIR
from src.logging_utils import log_event, setup_run_logger
from src.memory.service import MemoryService
from src.tools.registry import ToolArgumentError, ToolRegistry, UnknownToolError


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class AgentResult:
    run_id: str
    content: str
    stopped_reason: str


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
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_tool_calls = max_tool_calls
        self.memory_service = memory_service
        self.working_memory_max_messages = working_memory_max_messages
        self.working_memory_max_chars = working_memory_max_chars
        self.summary_trigger_messages = summary_trigger_messages
        self.messages: list[dict[str, str]] = []

    def run(self, user_input: str) -> AgentResult:
        run_id = uuid.uuid4().hex
        logger = setup_run_logger(run_id, LOGS_DIR)
        log_event(logger, "run_started", run_id=run_id, user_input=user_input)

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
                tool_result = self._execute_tool(parsed, logger)
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

    def _execute_tool(self, tool_call: ToolCall, logger: logging.Logger) -> str:
        try:
            result = self.tool_registry.execute(tool_call.tool, tool_call.arguments)
        except (UnknownToolError, ToolArgumentError) as exc:
            result = {"ok": False, "error": str(exc)}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        log_event(logger, "tool_result", tool=tool_call.tool, result=result)
        return json.dumps(result, ensure_ascii=False)

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
