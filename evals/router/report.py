from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_predictions(records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(metrics: dict[str, Any], records: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Router Benchmark Report",
        "",
        f"Mode: `{metrics['mode']}`",
        f"Dataset size: {metrics['total']}",
        f"Overall Accuracy: {metrics['accuracy']:.3f}",
        f"Macro F1: {metrics['macro_f1']:.3f}",
        f"LLM call count: {metrics['llm_call_count']}",
        f"LLM escalation rate: {metrics['llm_escalation_rate']:.3f}",
        "",
        "## Per-route Metrics",
        "",
        "| Route | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for route, item in sorted(metrics["per_route"].items()):
        lines.append(
            f"| `{route}` | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |"
        )

    labels = metrics["labels"]
    lines.extend(["", "## Confusion Matrix", "", "Rows are expected routes; columns are predicted routes.", ""])
    lines.append("| Expected \\ Predicted | " + " | ".join(f"`{label}`" for label in labels) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in labels) + " |")
    for expected in labels:
        row = metrics["confusion_matrix"].get(expected, {})
        lines.append("| `" + expected + "` | " + " | ".join(str(row.get(predicted, 0)) for predicted in labels) + " |")

    lines.extend(["", "## Category Accuracy", "", "| Category | Accuracy | Correct / Total |", "| --- | ---: | ---: |"])
    for category, item in sorted(metrics["category_accuracy"].items()):
        lines.append(f"| `{category}` | {item['accuracy']:.3f} | {item['correct']} / {item['total']} |")

    failures = [record for record in records if not record["correct"]]
    lines.extend(["", "## Representative Failures", ""])
    if not failures:
        lines.append("No failures.")
    else:
        for record in failures[:10]:
            lines.extend(
                [
                    f"### {record['id']}",
                    "",
                    f"- Input: {record['input']}",
                    f"- Expected: `{record['expected_route']}`",
                    f"- Predicted: `{record['predicted_route']}`",
                    f"- Category: `{record['category']}`",
                    f"- Reason: {record['reason']}",
                    "",
                ]
            )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
