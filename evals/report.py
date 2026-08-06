from __future__ import annotations

import time
from collections import Counter
from pathlib import Path


def compute_metrics(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    by_level: dict[int, dict] = {}
    for r in records:
        level = by_level.setdefault(r["level"], {"total": 0, "passed": 0})
        level["total"] += 1
        if r["passed"]:
            level["passed"] += 1
    route_dist = Counter(r["route"] for r in records)
    searched = sum(1 for r in records if r["search_count"] > 0)
    total_searches = sum(r["search_count"] for r in records)
    total_credits = sum(r["search_credits"] for r in records)
    total_llm = sum(r["llm_calls"] for r in records)
    citation_total = sum(r["citation_total"] for r in records)
    citation_ok = sum(r["citation_ok"] for r in records)
    failures = [r for r in records if not r["passed"]]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_level": by_level,
        "route_dist": route_dist,
        "searched": searched,
        "search_rate": searched / total if total else 0.0,
        "total_searches": total_searches,
        "total_credits": total_credits,
        "total_llm": total_llm,
        "citation_total": citation_total,
        "citation_ok": citation_ok,
        "failures": failures,
    }


def write_report(records: list[dict], out_dir: Path) -> None:
    m = compute_metrics(records)
    lines = [
        "# GAIA Evaluation Report",
        "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Questions: {m['total']}",
        f"- Passed: {m['passed']} ({m['pass_rate']:.1%})",
        f"- Searched at least once: {m['searched']}/{m['total']} ({m['search_rate']:.1%})",
        f"- Total searches: {m['total_searches']} (≈{m['total_credits']} Tavily credits)",
        f"- Total LLM calls: {m['total_llm']}",
        "",
        "## By level",
        "",
        "| Level | Total | Passed | Rate |",
        "|---|---|---|---|",
    ]
    for level in sorted(m["by_level"]):
        d = m["by_level"][level]
        rate = d["passed"] / d["total"] if d["total"] else 0.0
        lines.append(f"| {level} | {d['total']} | {d['passed']} | {rate:.1%} |")
    lines.append("")
    lines.append("## By route")
    lines.append("")
    for route, count in m["route_dist"].most_common():
        lines.append(f"- {route}: {count}")
    lines.append("")
    if m["citation_total"]:
        lines.append("## Citation honesty")
        lines.append("")
        lines.append(
            f"- Cited URLs checked: {m['citation_total']}, found in search results: "
            f"{m['citation_ok']} ({m['citation_ok'] / m['citation_total']:.1%})"
        )
        lines.append("")
    lines.append("## Failures")
    lines.append("")
    for r in m["failures"]:
        lines.append(
            f"- {r['task_id']} (L{r['level']}) route={r['route']} reason={r['stopped_reason'] or 'wrong_answer'} "
            f"expected={r['expected_answer']!r} extracted={r['extracted_answer']!r}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
