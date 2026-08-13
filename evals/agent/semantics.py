from __future__ import annotations

from typing import Any


FAILURE_STAGES = (
    "routing",
    "planning",
    "plan_validation",
    "argument_resolution",
    "tool_execution",
    "permission",
    "memory",
    "recovery",
    "final_answer",
    "runtime",
    "unknown",
)

_STAGE_ALIASES = {
    "": "",
    "none": "",
    "router": "routing",
    "runner": "runtime",
    "planner": "planning",
}


def canonical_failure_stage(stage: str | None) -> str:
    value = str(stage or "")
    canonical = _STAGE_ALIASES.get(value, value)
    if not canonical:
        return ""
    if canonical in FAILURE_STAGES:
        return canonical
    return "unknown"


def stage_for_status(record: dict[str, Any]) -> str:
    stopped_reason = str(record.get("stopped_reason") or "")
    final_status = str(record.get("final_status") or "")
    combined = f"{final_status} {stopped_reason}"
    if "confirmation" in combined or "policy" in combined:
        return "permission"
    if "planning_failed" in combined:
        return "planning"
    if "plan_validation" in combined or "validation" in combined:
        return "plan_validation"
    if "Argument resolver" in combined or "unresolved placeholders" in combined:
        return "argument_resolution"
    if "invalid_json" in combined or "model_parse_failed" in combined:
        return "recovery"
    if stopped_reason.startswith("llm_call_failed") or stopped_reason.startswith("runner_error"):
        return "runtime"
    if record.get("tool_execution_failures"):
        return "tool_execution"
    return "final_answer"


def tool_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    model_proposals = [
        str(event["tool"])
        for event in events
        if event.get("event") in ("model_tool_proposed", "proposed")
        and event.get("tool")
        and event.get("source") != "plan"
    ]
    planned_steps = [
        str(event["tool"])
        for event in events
        if event.get("event") in ("planned_tool_step", "proposed")
        and event.get("tool")
        and event.get("source") == "plan"
    ]
    legacy_proposals = [*model_proposals, *planned_steps]
    attempts = [str(event["tool"]) for event in events if event.get("event") == "execution_attempt" and event.get("tool")]
    successes = [str(event["tool"]) for event in events if event.get("event") == "execution_success" and event.get("tool")]
    failures = [event for event in events if event.get("event") == "execution_failure" and event.get("tool")]
    rejections = [str(event["tool"]) for event in events if event.get("event") == "policy_rejected" and event.get("tool")]
    return {
        "tool_events": events,
        "model_tool_proposals": model_proposals,
        "planned_tool_steps": planned_steps,
        "actual_tool_executions": attempts,
        "tool_proposals": legacy_proposals,
        "tool_calls": legacy_proposals,
        "tool_execution_attempts_by_name": attempts,
        "tool_execution_successes_by_name": successes,
        "tool_execution_failures_by_name": [str(event["tool"]) for event in failures],
        "tool_failures": failures,
        "tool_execution_attempts": len(attempts),
        "tool_execution_successes": len(successes),
        "tool_execution_failures": len(failures),
        "tool_policy_rejections": len(rejections),
    }
