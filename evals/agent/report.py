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
    route_correct = sum(1 for record in records if record.get("actual_route") == record.get("expected_route"))
    tool_calls = sum(len(record.get("tool_calls") or []) for record in records)
    tool_failures = sum(len(record.get("tool_failures") or []) for record in records)
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

    failure_stage_distribution = Counter(
        str(record.get("failure_stage") or "none") for record in records if not record.get("task_success")
    )

    return {
        "metadata": metadata or {},
        "total": total,
        "passed": passed,
        "overall_task_success_rate": passed / total if total else 0.0,
        "route_accuracy": route_correct / total if total else 0.0,
        "tool_execution_success_rate": (
            (tool_calls - tool_failures) / tool_calls if tool_calls else 1.0
        ),
        "average_tool_calls": tool_calls / total if total else 0.0,
        "average_llm_calls": llm_calls / total if total else 0.0,
        "retry_rate": retry_count / total if total else 0.0,
        "replan_rate": replan_count / total if total else 0.0,
        "total_tool_calls": tool_calls,
        "total_tool_failures": tool_failures,
        "total_llm_calls": llm_calls,
        "total_retry_count": retry_count,
        "total_replan_count": replan_count,
        "average_latency_ms": sum(latencies) / total if total else 0.0,
        "per_category": category_metrics,
        "failure_stage_distribution": dict(failure_stage_distribution),
        "route_distribution": dict(Counter(str(record.get("actual_route") or "") for record in records)),
        "representative_failures": [
            {
                "id": record.get("id"),
                "category": record.get("category"),
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
        "## Summary",
        "",
        f"- Overall task success: {metrics['passed']}/{metrics['total']} ({metrics['overall_task_success_rate']:.1%})",
        f"- Route accuracy: {metrics['route_accuracy']:.1%}",
        f"- Tool execution success rate: {metrics['tool_execution_success_rate']:.1%}",
        f"- Average tool calls: {metrics['average_tool_calls']:.2f}",
        f"- Average LLM calls: {metrics['average_llm_calls']:.2f}",
        f"- Retry rate: {metrics['retry_rate']:.2f}",
        f"- Replan rate: {metrics['replan_rate']:.2f}",
        f"- Average latency: {metrics['average_latency_ms']:.1f} ms",
        "",
        "## Per Category",
        "",
        "| Category | Total | Passed | Success Rate |",
        "|---|---:|---:|---:|",
    ]
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
                f"- {failure['id']} category={failure['category']} stage={failure['failure_stage']} "
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
