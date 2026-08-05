from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from src.config import MEMORY_DB_PATH
from src.memory.models import Memory, RetrievedMemory
from src.memory.store import MemoryStore


def tokenize_query(text: str) -> list[str]:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text.lower())
    return [token for token in tokens if token.strip()]


def query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in tokenize_query(text):
        terms.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            terms.extend(token[index : index + 2] for index in range(0, len(token) - 1))
    seen: set[str] = set()
    unique = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique


def build_fts_query(text: str) -> str:
    tokens = query_terms(text)
    return " OR ".join(f'"{token}"' for token in tokens[:12])


class MemoryRetriever:
    def __init__(self, store: MemoryStore | None = None, db_path: Path = MEMORY_DB_PATH) -> None:
        self.store = store or MemoryStore(db_path)

    def search(self, query: str, *, limit: int = 5) -> list[RetrievedMemory]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        try:
            rows = self._fts_search(fts_query, limit=max(limit * 4, limit))
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            rows = self._like_search(query, limit=max(limit * 4, limit))

        results: list[RetrievedMemory] = []
        for row in rows:
            memory = Memory.model_validate({**dict(row), "is_active": bool(row["is_active"])})
            fts_score = 1.0 / (1.0 + abs(float(row["rank_score"]))) if "rank_score" in row.keys() else 0.2
            score = fts_score + (memory.importance * 0.35) + self._recency_bonus(memory)
            results.append(RetrievedMemory(memory=memory, score=round(score, 6)))

        results.sort(key=lambda item: item.score, reverse=True)
        selected = results[:limit]
        self.store.touch_accessed([item.memory.id for item in selected])
        return selected

    def _fts_search(self, fts_query: str, *, limit: int) -> list[sqlite3.Row]:
        with self.store.connect() as conn:
            return conn.execute(
                """
                SELECT m.*, bm25(memories_fts) AS rank_score
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.is_active = 1
                ORDER BY rank_score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

    def _like_search(self, query: str, *, limit: int) -> list[sqlite3.Row]:
        terms = [term for term in query_terms(query) if len(term) >= 2][:12]
        if not terms:
            return []
        where = " OR ".join("LOWER(content) LIKE ?" for _ in terms)
        params = [f"%{term.lower()}%" for term in terms]
        with self.store.connect() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM memories
                WHERE is_active = 1 AND ({where})
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()

    def _recency_bonus(self, memory: Memory) -> float:
        # ISO dates sort lexically; this small stable bonus avoids recency dominating relevance.
        return 0.05 if memory.updated_at >= memory.created_at else 0.0
