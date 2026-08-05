from __future__ import annotations

import json
import uuid

from src.agent.runtime import AgentRuntime
from src.config import LOGS_DIR, load_settings
from src.logging_utils import log_event, setup_run_logger
from src.llm.client import LocalLLMClient
from src.memory.service import MemoryService
from src.memory.summarizer import ConversationSummarizer
from src.memory.journal import DevelopmentJournal
from src.planner.executor import PlanExecutor
from src.planner.models import PlanStatus, RiskLevel, Route
from src.planner.planner import StructuredPlanner
from src.planner.repository import PlanRepository
from src.planner.router import RequestRouter
from src.planner.validator import PlanValidator
from src.tools.calculator import CalculatorTool
from src.tools.file_tools import ReadTextFileTool, WriteTextFileTool
from src.tools.registry import ToolRegistry
from src.tools.transform_text import TransformTextTool


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(ReadTextFileTool())
    registry.register(WriteTextFileTool())
    registry.register(TransformTextTool())
    return registry


def main() -> None:
    settings = load_settings()
    llm_client = LocalLLMClient(
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
        model=settings.local_llm_model,
        timeout=settings.local_llm_timeout,
    )
    memory_service = MemoryService(
        summarizer=ConversationSummarizer(llm_client),
        retrieval_limit=settings.memory_retrieval_limit,
    )
    plan_repository = PlanRepository()
    router = RequestRouter()
    planner = StructuredPlanner(llm_client, build_registry(), max_steps=settings.planner_max_steps)
    validator = PlanValidator(build_registry(), max_steps=settings.planner_max_steps)
    runtime = AgentRuntime(
        llm_client=llm_client,
        tool_registry=build_registry(),
        max_tool_calls=settings.max_tool_calls,
        memory_service=memory_service,
        working_memory_max_messages=settings.working_memory_max_messages,
        working_memory_max_chars=settings.working_memory_max_chars,
        summary_trigger_messages=settings.summary_trigger_messages,
    )

    print("Local Agent Demo. Type /exit to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input == "/exit":
            break
        if not user_input:
            continue
        if handle_command(
            user_input,
            memory_service,
            plan_repository=plan_repository,
            executor_factory=lambda logger=None: PlanExecutor(
                registry=build_registry(),
                repository=plan_repository,
                llm_client=llm_client,
                validator=validator,
                max_retries=settings.tool_max_retries,
                max_replans=settings.planner_max_replans,
                confirm_write_actions=settings.confirm_write_actions,
                confirmation_callback=confirm_tool_action,
                logger=logger,
            ),
        ):
            continue

        memory_context = memory_service.retrieve_context(user_input, limit=settings.memory_retrieval_limit)
        route_decision = router.route(user_input, memory_context=memory_context.block)
        if route_decision.route == Route.CLARIFICATION:
            print("Assistant: " + " ".join(route_decision.missing_information))
            continue
        if route_decision.route == Route.PLANNED_TASK:
            run_id = uuid.uuid4().hex
            logger = setup_run_logger(run_id, LOGS_DIR)
            log_event(logger, "request_route", decision=route_decision.model_dump())
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
            try:
                plan = planner.create_plan(user_input, memory_context=memory_context.block)
            except Exception as exc:
                print(f"Assistant: 计划生成失败：{type(exc).__name__}: {exc}")
                continue
            validation = validator.validate(plan)
            log_event(logger, "original_plan", plan=plan.model_dump())
            log_event(logger, "plan_validation", ok=validation.ok, errors=validation.errors)
            if not validation.ok:
                print("Assistant: 计划校验失败：" + "; ".join(validation.errors))
                plan.status = PlanStatus.FAILED
                plan_repository.save(plan)
                continue
            plan_repository.save(plan)
            result = PlanExecutor(
                registry=build_registry(),
                repository=plan_repository,
                llm_client=llm_client,
                validator=validator,
                max_retries=settings.tool_max_retries,
                max_replans=settings.planner_max_replans,
                confirm_write_actions=settings.confirm_write_actions,
                confirmation_callback=confirm_tool_action,
                logger=logger,
            ).execute(plan)
            journal = DevelopmentJournal(memory_service).record_after_plan(result, source_run_id=run_id)
            if journal.saved:
                log_event(
                    logger,
                    "development_journal_saved",
                    memories=[{"id": memory.id, "content": memory.content} for memory in journal.saved],
                )
            if journal.skipped_reasons:
                log_event(logger, "development_journal_skipped", reasons=journal.skipped_reasons)
            print(f"Assistant: {result.final_answer}")
            print(f"[plan_id: {result.plan.plan_id}] [run_id: {run_id}]")
            continue

        result = runtime.run(user_input)
        print(f"Assistant: {result.content}")
        print(f"[run_id: {result.run_id}]")


def handle_command(user_input: str, memory_service: MemoryService, *, plan_repository: PlanRepository, executor_factory) -> bool:
    if user_input.startswith("/remember "):
        content = user_input.removeprefix("/remember ").strip()
        if not content:
            print("Usage: /remember <内容>")
            return True
        memory = memory_service.remember_explicit(content)
        print(f"Remembered [{memory.id}]: {memory.content}")
        return True

    if user_input == "/memories":
        memories = memory_service.list_memories()
        if not memories:
            print("No active memories.")
            return True
        for memory in memories:
            print(f"{memory.id}\t{memory.category}\t{memory.created_at}\t{memory.content}")
        return True

    if user_input.startswith("/forget "):
        raw_id = user_input.removeprefix("/forget ").strip()
        try:
            memory_service.forget(int(raw_id))
        except (ValueError, KeyError) as exc:
            print(f"Could not forget memory: {exc}")
            return True
        print(f"Forgot memory {raw_id}.")
        return True

    if user_input.startswith("/update-memory "):
        rest = user_input.removeprefix("/update-memory ").strip()
        parts = rest.split(maxsplit=1)
        if len(parts) != 2:
            print("Usage: /update-memory <memory_id> <新内容>")
            return True
        raw_id, content = parts
        try:
            memory = memory_service.update_memory(int(raw_id), content)
        except (ValueError, KeyError) as exc:
            print(f"Could not update memory: {exc}")
            return True
        print(f"Updated [{memory.id}]: {memory.content}")
        return True

    if user_input == "/clear-session":
        memory_service.clear_session()
        print("Cleared current session working memory and summary.")
        return True

    if user_input == "/summary":
        print(memory_service.conversation_summary or "No conversation summary yet.")
        return True

    if user_input == "/plans":
        plans = plan_repository.list()
        if not plans:
            print("No plans.")
            return True
        for plan in plans:
            print(f"{plan.plan_id}\t{plan.status}\t{plan.updated_at}\t{plan.goal}")
        return True

    if user_input.startswith("/plan "):
        plan_id = user_input.removeprefix("/plan ").strip()
        plan = plan_repository.get(plan_id)
        if plan is None:
            print(f"Plan not found: {plan_id}")
            return True
        print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
        return True

    if user_input.startswith("/cancel-plan "):
        plan_id = user_input.removeprefix("/cancel-plan ").strip()
        plan = plan_repository.get(plan_id)
        if plan is None:
            print(f"Plan not found: {plan_id}")
            return True
        plan.status = PlanStatus.CANCELLED
        plan_repository.save(plan)
        print(f"Cancelled plan {plan_id}.")
        return True

    if user_input.startswith("/resume-plan "):
        plan_id = user_input.removeprefix("/resume-plan ").strip()
        plan = plan_repository.get(plan_id)
        if plan is None:
            print(f"Plan not found: {plan_id}")
            return True
        if plan.status == PlanStatus.CANCELLED:
            print(f"Plan {plan_id} is cancelled and cannot be resumed.")
            return True
        run_id = uuid.uuid4().hex
        logger = setup_run_logger(run_id, LOGS_DIR)
        plan.status = PlanStatus.RUNNING
        result = executor_factory(logger).execute(plan)
        print(f"Assistant: {result.final_answer}")
        print(f"[plan_id: {result.plan.plan_id}] [run_id: {run_id}]")
        return True

    if user_input == "/history":
        plans = plan_repository.list()
        memories = memory_service.list_memories()
        print(f"Plans: {len(plans)}")
        for plan in plans[:10]:
            print(f"{plan.plan_id}\t{plan.status}\t{plan.updated_at}\t{plan.goal}")
        print(f"Active memories: {len(memories)}")
        return True

    return False


def confirm_tool_action(plan, step, risk: RiskLevel, reason: str) -> bool:
    print(f"Confirm {risk} action for plan {plan.plan_id}, step {step.id}: {reason}")
    print(f"Tool: {step.tool_name}")
    print(f"Arguments: {json.dumps(step.arguments, ensure_ascii=False)}")
    answer = input("Proceed? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


if __name__ == "__main__":
    main()
