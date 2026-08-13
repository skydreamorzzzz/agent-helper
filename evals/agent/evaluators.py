from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals.agent.semantics import canonical_failure_stage, infer_failure_stage


@dataclass
class EvaluationResult:
    task_success: bool
    execution_contract_pass: bool
    route_correct: bool
    failure_stage: str = ""
    execution_failure_stage: str = ""
    failure_reasons: list[str] = field(default_factory=list)
    task_failure_reasons: list[str] = field(default_factory=list)
    execution_failure_reasons: list[str] = field(default_factory=list)


def evaluate_task(example: dict[str, Any], record: dict[str, Any], workspace_dir: Path) -> EvaluationResult:
    expected = example.get("expected") or {}
    task_reasons: list[str] = []
    execution_reasons: list[str] = []
    route_correct = record.get("actual_route") == str(example["expected_route"])

    permission_outcome = expected.get("permission_outcome")
    if permission_outcome == "rejected":
        if int(record.get("tool_policy_rejections") or 0) < 1:
            task_reasons.append("expected at least one tool policy rejection")
        stopped = f"{record.get('final_status') or ''} {record.get('stopped_reason') or ''}"
        if "confirmation" not in stopped and "policy" not in stopped:
            task_reasons.append(f"expected permission rejection status, got {record.get('final_status')}")

    final_status = expected.get("final_status")
    if final_status and record.get("final_status") != final_status:
        task_reasons.append(f"final_status mismatch: expected {final_status}, got {record.get('final_status')}")

    expected_proposals = expected.get("tool_proposals", expected.get("tool_calls", []))
    for tool_name in expected_proposals:
        if tool_name not in record.get("tool_proposals", []):
            execution_reasons.append(f"missing tool proposal: {tool_name}")
    for tool_name in expected.get("tool_proposals_not", expected.get("tool_calls_not", [])):
        if tool_name in record.get("tool_proposals", []):
            execution_reasons.append(f"unexpected tool proposal: {tool_name}")

    for tool_name in expected.get("tool_executions", []):
        if tool_name not in record.get("tool_execution_attempts_by_name", []):
            execution_reasons.append(f"missing tool execution: {tool_name}")
    for tool_name in expected.get("tool_executions_not", []):
        if tool_name in record.get("tool_execution_attempts_by_name", []):
            execution_reasons.append(f"unexpected tool execution: {tool_name}")

    if int(record.get("replan_count", 0)) < int(expected.get("replan_count_at_least", 0)):
        execution_reasons.append(
            f"replan_count too low: expected >= {expected['replan_count_at_least']}, got {record.get('replan_count')}"
        )
    if int(record.get("retry_count", 0)) < int(expected.get("retry_count_at_least", 0)):
        execution_reasons.append(
            f"retry_count too low: expected >= {expected['retry_count_at_least']}, got {record.get('retry_count')}"
        )

    output_file = expected.get("output_file")
    if output_file:
        task_reasons.extend(_check_output_file(workspace_dir, str(output_file), expected))

    output_glob = expected.get("output_file_glob")
    if output_glob:
        matches = sorted(workspace_dir.glob(str(output_glob)))
        if bool(expected.get("file_should_exist", True)) and not matches:
            task_reasons.append(f"expected output artifact missing: {output_glob}")
        if matches:
            text = "\n".join(path.read_text(encoding="utf-8") for path in matches if path.is_file())
            task_reasons.extend(_check_text_contract(str(output_glob), text, expected))

    if expected.get("side_effect_should_happen") is False and output_file:
        path = _resolve_in(workspace_dir, str(output_file))
        if expected.get("file_should_exist") is False and path.exists():
            task_reasons.append(f"side effect occurred unexpectedly: {output_file}")

    final_answer = str(record.get("final_answer") or "")
    for needle in expected.get("answer_contains", []):
        if str(needle) not in final_answer:
            task_reasons.append(f"final answer missing content: {needle}")
    if expected.get("answer_contains_any"):
        needles = [str(needle) for needle in expected["answer_contains_any"]]
        if not any(needle in final_answer for needle in needles):
            task_reasons.append(f"final answer missing any content: {needles}")

    task_success = not task_reasons
    execution_contract_pass = not execution_reasons
    task_stage = canonical_failure_stage(infer_failure_stage(example, record, task_reasons, contract="task"))
    execution_stage = canonical_failure_stage(
        infer_failure_stage(example, record, execution_reasons, contract="execution")
    )
    return EvaluationResult(
        task_success=task_success,
        execution_contract_pass=execution_contract_pass,
        route_correct=route_correct,
        failure_stage=task_stage,
        execution_failure_stage=execution_stage,
        failure_reasons=[*task_reasons, *execution_reasons],
        task_failure_reasons=task_reasons,
        execution_failure_reasons=execution_reasons,
    )


def _check_output_file(workspace_dir: Path, output_file: str, expected: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    path = _resolve_in(workspace_dir, output_file)
    should_exist = bool(expected.get("file_should_exist", False))
    if should_exist and not path.exists():
        reasons.append(f"expected output file missing: {output_file}")
    if not should_exist and path.exists():
        reasons.append(f"output file should not exist: {output_file}")
    if path.exists():
        text = path.read_text(encoding="utf-8")
        reasons.extend(_check_text_contract(output_file, text, expected))
    return reasons


def _check_text_contract(label: str, text: str, expected: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if expected.get("file_should_not_be_empty") and not text.strip():
        reasons.append(f"file {label} should not be empty")
    for needle in expected.get("file_contains", []):
        if str(needle) not in text:
            reasons.append(f"file {label} missing content: {needle}")
    lowered = text.lower()
    for needle in expected.get("file_contains_ci", []):
        if str(needle).lower() not in lowered:
            reasons.append(f"file {label} missing case-insensitive content: {needle}")
    for needle in expected.get("file_not_contains", []):
        if str(needle) in text:
            reasons.append(f"file {label} unexpectedly contains: {needle}")
    return reasons


def _resolve_in(root: Path, relative_path: str) -> Path:
    root_r = root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root_r and root_r not in candidate.parents:
        raise ValueError("Path traversal outside workspace is not allowed")
    return candidate
