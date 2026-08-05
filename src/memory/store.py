from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from src.config import MEMORY_DB_PATH
from src.memory.models import Memory, MemoryCreate, utc_now_iso


class MemoryStore:
    def __init__(self, db_path: Path = MEMORY_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL CHECK (category IN (
                        'user_preference',
                        'personal_fact',
                        'project',
                        'task',
                        'decision',
                        'summary'
                    )),
                    content TEXT NOT NULL,
                    source_run_id TEXT NOT NULL,
                    source_message_id TEXT NOT NULL,
                    importance REAL NOT NULL CHECK (importance >= 0 AND importance <= 1),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, category, content='memories', content_rowid='id')
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, category)
                    VALUES('delete', old.id, old.content, old.category);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content, category)
                    VALUES('delete', old.id, old.content, old.category);
                    INSERT INTO memories_fts(rowid, content, category)
                    VALUES (new.id, new.content, new.category);
                END
                """
            )

    def add(self, memory: MemoryCreate) -> Memory:
        now = utc_now_iso()
        metadata_json = json.dumps(memory.metadata, ensure_ascii=False)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    category, content, source_run_id, source_message_id, importance,
                    created_at, updated_at, is_active, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    str(memory.category),
                    memory.content.strip(),
                    memory.source_run_id,
                    memory.source_message_id,
                    memory.importance,
                    now,
                    now,
                    metadata_json,
                ),
            )
            memory_id = int(cursor.lastrowid)
        found = self.get(memory_id)
        if found is None:
            raise RuntimeError("Inserted memory could not be loaded")
        return found

    def get(self, memory_id: int, *, include_inactive: bool = False) -> Memory | None:
        clause = "" if include_inactive else " AND is_active = 1"
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM memories WHERE id = ?{clause}",
                (memory_id,),
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_active(self) -> list[Memory]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE is_active = 1 ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def update(self, memory_id: int, content: str) -> Memory:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, updated_at = ?
                WHERE id = ? AND is_active = 1
                """,
                (content.strip(), now, memory_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Active memory not found: {memory_id}")
        found = self.get(memory_id)
        if found is None:
            raise RuntimeError("Updated memory could not be loaded")
        return found

    def soft_delete(self, memory_id: int) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET is_active = 0, updated_at = ?
                WHERE id = ? AND is_active = 1
                """,
                (now, memory_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Active memory not found: {memory_id}")

    def touch_accessed(self, memory_ids: Iterable[int]) -> None:
        ids = list(memory_ids)
        if not ids:
            return
        now = utc_now_iso()
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE memories SET last_accessed_at = ? WHERE id IN ({placeholders})",
                (now, *ids),
            )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        data = dict(row)
        data["is_active"] = bool(data["is_active"])
        return Memory.model_validate(data)

