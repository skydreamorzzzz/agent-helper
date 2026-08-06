"""GAIA benchmark data loading and official-style rule-based grading.

Data is fetched from the ModelScope mirror of gaia-benchmark/GAIA (no HF token
needed). The mirror stores validation metadata as parquet files. Grading follows
the official GAIA scorer: numbers are parsed and compared exactly, lists are
compared element-wise, strings are normalized.
"""

from __future__ import annotations

import io
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow.parquet as pq

MODELSCOPE_DATASET = "gaia-benchmark/GAIA"
MODELSCOPE_BASE = "https://www.modelscope.cn/api/v1/datasets"
_PARQUET_PATH = "2023/validation/metadata.level{level}.parquet"
_COLUMN_MAP = {"Question": "question", "Final answer": "answer", "Level": "level"}


@dataclass(frozen=True)
class GaiaQuestion:
    task_id: str
    question: str
    level: int
    answer: str


def _modelscope_url(file_path: str) -> str:
    return f"{MODELSCOPE_BASE}/{MODELSCOPE_DATASET}/repo?Revision=master&FilePath={file_path}"


def _read_parquet(source: io.BytesIO | Path) -> list[dict]:
    table = pq.read_table(source)
    columns = table.column_names
    rows: list[dict] = []
    for batch in table.to_batches():
        arrays = {column: batch.column(column).to_pylist() for column in columns}
        for i in range(batch.num_rows):
            rows.append({_COLUMN_MAP.get(column, column): arrays[column][i] for column in columns})
    return rows


def _parse_metadata_line(line: str) -> dict:
    return json.loads(line)


def _read_metadata_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".parquet":
        return _read_parquet(path)
    return [_parse_metadata_line(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fetch_level(level: int) -> list[dict] | None:
    try:
        response = httpx.get(_modelscope_url(_PARQUET_PATH.format(level=level)), timeout=60.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        return _read_parquet(io.BytesIO(response.content))
    except httpx.HTTPError:
        return None


def _row_level(row: dict) -> int | None:
    level = row.get("level")
    if level is None:
        return None
    try:
        return int(level)
    except (TypeError, ValueError):
        return None


def fetch_gaia_metadata(levels: tuple[int, ...] = (1, 2), cache_dir: Path | None = None) -> dict[int, list[dict]]:
    metadata: dict[int, list[dict]] = {}
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        cache_file = cache_dir / f"metadata_l{level}.json" if cache_dir else None
        if cache_file and cache_file.exists():
            metadata[level] = json.loads(cache_file.read_text(encoding="utf-8"))
            continue
        rows = _fetch_level(level)
        if rows is None:
            raise RuntimeError(
                f"Failed to download GAIA level {level} metadata from ModelScope. "
                "Provide a local metadata file instead (see run_eval.py --metadata)."
            )
        metadata[level] = rows
        if cache_file:
            cache_file.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_gaia_metadata_from_files(level_paths: dict[int, Path]) -> dict[int, list[dict]]:
    return {level: _read_metadata_file(path) for level, path in level_paths.items()}


def select_questions(
    metadata: dict[int, list[dict]],
    levels: tuple[int, ...] = (1, 2),
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[GaiaQuestion]:
    pool: list[GaiaQuestion] = []
    for level in levels:
        for row in metadata.get(level, []):
            if row.get("file_name"):
                continue  # attachment questions are out of scope (no download/vision tools)
            if _row_level(row) not in levels:
                continue
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            pool.append(GaiaQuestion(task_id=str(row.get("task_id")), question=question, level=level, answer=answer))
    pool.sort(key=lambda q: q.task_id)
    if limit is not None and len(pool) > limit:
        return random.Random(seed).sample(pool, limit)
    return pool


def normalize_answer(s: str) -> str:
    text = unicodedata.normalize("NFKD", s)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9一-鿿 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article):]
            break
    return text


def _to_float(s: str) -> float | None:
    text = s.strip().replace("$", "").replace("%", "").replace("±", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _split_list(s: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;]", s) if part.strip()]


def is_correct(predicted: str, answer: str) -> bool:
    p = (predicted or "").strip()
    a = (answer or "").strip()
    if not p or not a:
        return False
    if p == a:
        return True
    if "," in a or ";" in a:
        ap = _split_list(a)
        pp = _split_list(p)
        if ap and pp and len(ap) == len(pp):
            if set(normalize_answer(x) for x in ap) == set(normalize_answer(x) for x in pp):
                return True
    af = _to_float(a)
    pf = _to_float(p)
    if af is not None and pf is not None and af == pf:
        return True
    return normalize_answer(p) == normalize_answer(a)
