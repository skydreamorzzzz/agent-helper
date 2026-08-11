from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FAILURE_STAGES = {
    "routing",
    "planning",
    "argument_resolution",
    "tool_execution",
    "permission",
    "recovery",
    "final_answer",
    "memory",
    "runner",
}


@dataclass
class EvaluationResult:
    task_success: bool
    route_correct: bool
    failure_stage: str = ""
    failure_reasons: list[str] = field(default_factory=list)


def evaluate_task(example: dict[str, Any], record: dict[str, Any], workspace_dir: Path) -> EvaluationResult:
    expected = example.get("expected") or {}
    reasons: list[str] = []
    route_correct = record.get("actual_route") == str(example["expected_route"])

    permission_outcome = expected.get("permission_outcome")
    if permission_outcome == "rejected":
        if int(record.get("tool_policy_rejections") or 0) < 1:
            reasons.append("expected at least one tool policy rejection")
        stopped = f"{record.get('final_status') or ''} {record.get('stopped_reason') or ''}"
        if "confirmation" not in stopped and "policy" not in stopped:
            reasons.append(f"expected permission rejection status, got {record.get('final_status')}")

    final_status = expected.get("final_status")
    if final_status and record.get("final_status") != final_status:
        reasons.append(f"final_status mismatch: expected {final_status}, got {record.get('final_status')}")

    expected_proposals = expected.get("tool_proposals", expected.get("tool_calls", []))
    for tool_name in expected_proposals:
        if tool_name not in record.get("tool_proposals", []):
            reasons.append(f"missing tool proposal: {tool_name}")
    for tool_name in expected.get("tool_proposals_not", expected.get("tool_calls_not", [])):
        if tool_name in record.get("tool_proposals", []):
            reasons.append(f"unexpected tool proposal: {tool_name}")

    for tool_name in expected.get("tool_executions", []):
        if tool_name not in record.get("tool_execution_attempts_by_name", []):
            reasons.append(f"missing tool execution: {tool_name}")
    for tool_name in expected.get("tool_executions_not", []):
        if tool_name in record.get("tool_execution_attempts_by_name", []):
            reasons.append(f"unexpected tool execution: {tool_name}")

    if int(record.get("replan_count", 0)) < int(expected.get("replan_count_at_least", 0)):
        reasons.append(
            f"replan_count too low: expected >= {expected['replan_count_at_least']}, got {record.get('replan_count')}"
        )
    if int(record.get("retry_count", 0)) < int(expected.get("retry_count_at_least", 0)):
        reasons.append(
            f"retry_count too low: expected >= {expected['retry_count_at_least']}, got {record.get('retry_count')}"
        )

    output_file = expected.get("output_file")
    if output_file:
        path = _resolve_in(workspace_dir, str(output_file))
        should_exist = bool(expected.get("file_should_exist", False))
        if should_exist and not path.exists():
            reasons.append(f"expected output file missing: {output_file}")
        if not should_exist and path.exists():
            reasons.append(f"output file should not exist: {output_file}")
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for needle in expected.get("file_contains", []):
                if str(needle) not in text:
                    reasons.append(f"file {output_file} missing content: {needle}")
            for needle in expected.get("file_not_contains", []):
                if str(needle) in text:
                    reasons.append(f"file {output_file} unexpectedly contains: {needle}")

    if expected.get("side_effect_should_happen") is False and output_file:
        path = _resolve_in(workspace_dir, str(output_file))
        if expected.get("file_should_exist") is False and path.exists():
            reasons.append(f"side effect occurred unexpectedly: {output_file}")

    final_answer = str(record.get("final_answer") or "")
    for needle in expected.get("answer_contains", []):
        if str(needle) not in final_answer:
            reasons.append(f"final answer missing content: {needle}")

    if reasons:
        return EvaluationResult(False, route_correct, _infer_failure_stage(example, record, reasons), reasons)
    return EvaluationResult(True, route_correct)


def _resolve_in(root: Path, relative_path: str) -> Path:
    root_r = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root_r and root_r not in candidate.parents:
        raise ValueError("Path traversal outside workspace is not allowed")
    return candidate


def _stage_for_status(record: dict[str, Any]) -> str:
    stopped_reason = str(record.get("stopped_reason") or "")
    final_status = str(record.get("final_status") or "")
    if "confirmation" in stopped_reason or "confirmation" in final_status:
        return "permission"
    if "plan" in stopped_reason or "validation" in stopped_reason:
        return "planning"
    if "invalid_json" in stopped_reason:
        return "recovery"
    if record.get("tool_execution_failures"):
        return "tool_execution"
    if stopped_reason.startswith("runner_error"):
        return "runner"
    return "final_answer"


def _infer_failure_stage(example: dict[str, Any], record: dict[str, Any], reasons: list[str]) -> str:
    category = str(example.get("category") or "")
    if "memory" in category:
        return "memory"
    if any("route mismatch" in reason for reason in reasons):
        return "routing"
    if any("tool proposal" in reason or "tool execution" in reason for reason in reasons):
        return "tool_execution"
    if any("file" in reason for reason in reasons):
        if "confirmation" in str(record.get("stopped_reason") or ""):
            return "permission"
        return "tool_execution"
    if record.get("tool_execution_failures"):
        return "tool_execution"
    return _stage_for_status(record)
