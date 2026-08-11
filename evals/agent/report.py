from __future__ import annotations

import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def compute_metrics(records: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record.get("task_success"))
    route_correct = sum(1 for record in records if record.get("route_correct"))
    tool_proposals = sum(len(record.get("tool_proposals") or []) for record in records)
    tool_execution_attempts = sum(int(record.get("tool_execution_attempts") or 0) for record in records)
    tool_execution_successes = sum(int(record.get("tool_execution_successes") or 0) for record in records)
    tool_execution_failures = sum(int(record.get("tool_execution_failures") or 0) for record in records)
    tool_policy_rejections = sum(int(record.get("tool_policy_rejections") or 0) for record in records)
    llm_calls = sum(int(record.get("llm_calls") or 0) for record in records)
    retry_count = sum(int(record.get("retry_count") or 0) for record in records)
    replan_count = sum(int(record.get("replan_count") or 0) for record in records)
    latencies = [float(record.get("latency_ms") or 0.0) for record in records]

    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "passed": 0})
    for record in records:
        item = by_category[str(record.get("category") or "unknown")]
        item["total"] += 1
        if record.get("task_success"):
            item["passed"] += 1
    category_metrics = {
        category: {
            "total": item["total"],
            "passed": item["passed"],
            "success_rate": item["passed"] / item["total"] if item["total"] else 0.0,
        }
        for category, item in sorted(by_category.items())
    }

    suite_metrics: dict[str, dict[str, Any]] = {}
    for suite in sorted({str(record.get("suite") or "normal") for record in records}):
        suite_records = [record for record in records if str(record.get("suite") or "normal") == suite]
        suite_total = len(suite_records)
        suite_passed = sum(1 for record in suite_records if record.get("task_success"))
        suite_metrics[suite] = {
            "total": suite_total,
            "passed": suite_passed,
            "pass_rate": suite_passed / suite_total if suite_total else 0.0,
        }

    failure_stage_distribution = Counter(
        str(record.get("failure_stage") or "none") for record in records if not record.get("task_success")
    )

    return {
        "metadata": metadata or {},
        "total": total,
        "passed": passed,
        "overall_integration_pass_rate": passed / total if total else 0.0,
        "overall_task_success_rate": passed / total if total else 0.0,
        "normal_task_success_rate": suite_metrics.get("normal", {}).get("pass_rate", 0.0),
        "regression_case_pass_rate": suite_metrics.get("regression", {}).get("pass_rate", 0.0),
        "route_accuracy": route_correct / total if total else 0.0,
        "tool_execution_success_rate": (
            tool_execution_successes / (tool_execution_successes + tool_execution_failures)
            if tool_execution_successes + tool_execution_failures
            else 1.0
        ),
        "average_tool_proposals": tool_proposals / total if total else 0.0,
        "average_tool_calls": tool_proposals / total if total else 0.0,
        "average_llm_calls": llm_calls / total if total else 0.0,
        "retry_rate": retry_count / total if total else 0.0,
        "replan_rate": replan_count / total if total else 0.0,
        "tool_proposals": tool_proposals,
        "tool_execution_attempts": tool_execution_attempts,
        "tool_execution_successes": tool_execution_successes,
        "tool_execution_failures": tool_execution_failures,
        "tool_policy_rejections": tool_policy_rejections,
        "total_tool_calls": tool_proposals,
        "total_tool_failures": tool_execution_failures,
        "total_llm_calls": llm_calls,
        "total_retry_count": retry_count,
        "total_replan_count": replan_count,
        "average_latency_ms": sum(latencies) / total if total else 0.0,
        "per_suite": suite_metrics,
        "per_category": category_metrics,
        "failure_stage_distribution": dict(failure_stage_distribution),
        "route_distribution": dict(Counter(str(record.get("actual_route") or "") for record in records)),
        "representative_failures": [
            {
                "id": record.get("id"),
                "category": record.get("category"),
                "suite": record.get("suite"),
                "failure_stage": record.get("failure_stage"),
                "reasons": record.get("failure_reasons"),
                "actual_route": record.get("actual_route"),
                "stopped_reason": record.get("stopped_reason"),
            }
            for record in records
            if not record.get("task_success")
        ][:8],
    }


def write_outputs(records: list[dict[str, Any]], out_dir: Path, *, metadata: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(records, metadata=metadata)
    (out_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(metrics), encoding="utf-8")
    return metrics


def render_markdown(metrics: dict[str, Any]) -> str:
    metadata = metrics.get("metadata") or {}
    lines = [
        "# End-to-End Agent Evaluation Report",
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Mode: {metadata.get('mode', 'unknown')}",
        f"- Dataset version: {metadata.get('dataset_version', 'unknown')}",
        f"- Git commit: {metadata.get('git_commit', 'unknown')}",
        f"- LLM model: {metadata.get('llm_model', 'unknown')}",
        f"- Router configuration: {metadata.get('router_configuration', 'current RequestRouter; router tuning frozen')}",
        "",
        "> This deterministic integration benchmark uses a fake model and fake web search. "
        "It measures system integration contracts, not live LLM agent quality or real user task success.",
        "",
        "## Summary",
        "",
        f"- Overall integration pass rate: {metrics['passed']}/{metrics['total']} ({metrics['overall_integration_pass_rate']:.1%})",
        f"- Normal task pass rate: {metrics['normal_task_success_rate']:.1%}",
        f"- Regression case pass rate: {metrics['regression_case_pass_rate']:.1%}",
        f"- Route accuracy: {metrics['route_accuracy']:.1%}",
        f"- Tool proposals: {metrics['tool_proposals']}",
        f"- Tool execution attempts: {metrics['tool_execution_attempts']}",
        f"- Tool execution successes: {metrics['tool_execution_successes']}",
        f"- Tool execution failures: {metrics['tool_execution_failures']}",
        f"- Tool policy rejections: {metrics['tool_policy_rejections']}",
        f"- Tool execution success rate: {metrics['tool_execution_success_rate']:.1%}",
        f"- Average tool proposals: {metrics['average_tool_proposals']:.2f}",
        f"- Average LLM calls: {metrics['average_llm_calls']:.2f}",
        f"- Retry rate: {metrics['retry_rate']:.2f}",
        f"- Replan rate: {metrics['replan_rate']:.2f}",
        f"- Average latency: {metrics['average_latency_ms']:.1f} ms",
        "",
        "## Per Suite",
        "",
        "| Suite | Total | Passed | Pass Rate |",
        "|---|---:|---:|---:|",
    ]
    for suite, item in metrics["per_suite"].items():
        lines.append(f"| {suite} | {item['total']} | {item['passed']} | {item['pass_rate']:.1%} |")
    lines.extend([
        "",
        "## Per Category",
        "",
        "| Category | Total | Passed | Success Rate |",
        "|---|---:|---:|---:|",
    ])
    for category, item in metrics["per_category"].items():
        lines.append(f"| {category} | {item['total']} | {item['passed']} | {item['success_rate']:.1%} |")
    lines.extend(["", "## Failure Stages", ""])
    if metrics["failure_stage_distribution"]:
        for stage, count in sorted(metrics["failure_stage_distribution"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Route Distribution", ""])
    for route, count in sorted(metrics["route_distribution"].items()):
        lines.append(f"- {route}: {count}")
    lines.extend(["", "## Representative Failures", ""])
    if metrics["representative_failures"]:
        for failure in metrics["representative_failures"]:
            reasons = "; ".join(str(reason) for reason in failure.get("reasons") or [])
            lines.append(
                f"- {failure['id']} suite={failure['suite']} category={failure['category']} stage={failure['failure_stage']} "
                f"route={failure['actual_route']} reason={failure['stopped_reason']}: {reasons}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
