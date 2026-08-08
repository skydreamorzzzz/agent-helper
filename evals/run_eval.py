from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from src.cli import build_lookup_registry, build_research_registry
from src.config import load_settings
from src.llm.client import LocalLLMClient

from evals.gaia import (
    fetch_gaia_metadata,
    load_gaia_metadata_from_files,
    select_questions,
)
from evals.report import write_report
from evals.runner import QuestionRecord, run_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA evaluation on the local agent.")
    parser.add_argument("--limit", type=int, default=30, help="max questions to run (default 30)")
    parser.add_argument("--levels", default="1,2", help="comma-separated levels, e.g. 1,2")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for reproducibility")
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=[],
        help="local metadata files as level:path, e.g. 1:evals/data/meta_l1.jsonl 2:evals/data/meta_l2.jsonl",
    )
    parser.add_argument("--out", default="evals/results", help="output directory")
    args = parser.parse_args()

    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    settings = load_settings()
    llm_client = LocalLLMClient(
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key,
        model=settings.local_llm_model,
        timeout=settings.local_llm_timeout,
    )
    lookup_registry = build_lookup_registry(settings.tavily_api_key)
    research_registry = build_research_registry(settings.tavily_api_key)

    data_dir = Path("evals/data")
    if args.metadata:
        paths: dict[int, Path] = {}
        for item in args.metadata:
            level_str, _, path = item.partition(":")
            paths[int(level_str)] = Path(path)
        metadata = load_gaia_metadata_from_files(paths)
    else:
        metadata = fetch_gaia_metadata(levels=levels, cache_dir=data_dir)

    questions = select_questions(metadata, levels=levels, limit=args.limit, seed=args.seed)
    print(f"Loaded {len(questions)} GAIA questions (levels {levels}).")
    if not questions:
        raise SystemExit("No questions selected.")

    out_dir = Path(args.out) / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_task_ids.json").write_text(
        json.dumps([q.task_id for q in questions], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="gaia_eval_") as tmp:
        workspace_root = Path(tmp)
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q.task_id} (L{q.level})", flush=True)
            try:
                rec = run_question(
                    q,
                    llm_client=llm_client,
                    lookup_registry=lookup_registry,
                    research_registry=research_registry,
                    settings=settings,
                    workspace_root=workspace_root,
                )
            except Exception as exc:
                rec = QuestionRecord(task_id=q.task_id, question=q.question, level=q.level, expected_answer=q.answer)
                rec.stopped_reason = f"runner_error:{type(exc).__name__}:{exc}"
            records.append(rec.to_dict())
            with (out_dir / "questions.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            print(
                f"  route={rec.route} pass={rec.passed} searches={rec.search_count} llm={rec.llm_calls}",
                flush=True,
            )

    write_report(records, out_dir)
    print(f"Done. Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
