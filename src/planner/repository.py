from __future__ import annotations

import sqlite3
from pathlib import Path

from src.config import PLAN_DB_PATH
from src.memory.models import utc_now_iso
from src.planner.models import Plan, PlanStatus


class PlanRepository:
    def __init__(self, db_path: Path = PLAN_DB_PATH) -> None:
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
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    plan_json TEXT NOT NULL
                )
                """
            )

    def save(self, plan: Plan) -> Plan:
        plan.updated_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO plans(plan_id, status, goal, created_at, updated_at, plan_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    status = excluded.status,
                    goal = excluded.goal,
                    updated_at = excluded.updated_at,
                    plan_json = excluded.plan_json
                """,
                (
                    plan.plan_id,
                    str(plan.status),
                    plan.goal,
                    plan.created_at,
                    plan.updated_at,
                    plan.model_dump_json(),
                ),
            )
        return plan

    def get(self, plan_id: str) -> Plan | None:
        with self.connect() as conn:
            row = conn.execute("SELECT plan_json FROM plans WHERE plan_id = ?", (plan_id,)).fetchone()
        return Plan.model_validate_json(row["plan_json"]) if row else None

    def list(self) -> list[Plan]:
        with self.connect() as conn:
            rows = conn.execute("SELECT plan_json FROM plans ORDER BY updated_at DESC").fetchall()
        return [Plan.model_validate_json(row["plan_json"]) for row in rows]

    def list_incomplete(self) -> list[Plan]:
        incomplete = {
            PlanStatus.PENDING,
            PlanStatus.RUNNING,
            PlanStatus.PAUSED,
            PlanStatus.FAILED,
        }
        return [plan for plan in self.list() if plan.status in incomplete]

