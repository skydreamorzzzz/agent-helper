from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
LOGS_DIR = PROJECT_ROOT / "logs"
MEMORY_DB_PATH = WORKSPACE_DIR / "memory.sqlite3"
PLAN_DB_PATH = WORKSPACE_DIR / "plans.sqlite3"


@dataclass(frozen=True)
class Settings:
    local_llm_base_url: str
    local_llm_api_key: str
    local_llm_model: str
    local_llm_timeout: float = 600.0
    max_tool_calls: int = 5
    working_memory_max_messages: int = 12
    working_memory_max_chars: int = 12000
    summary_trigger_messages: int = 16
    memory_retrieval_limit: int = 5
    memory_importance_threshold: float = 0.6
    planner_max_steps: int = 8
    planner_max_replans: int = 2
    tool_max_retries: int = 1
    confirm_write_actions: bool = True
    tavily_api_key: str = ""


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")

    missing = [
        name
        for name in ("LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL")
        if not os.getenv(name)
    ]
    api_key = _load_api_key()
    if not api_key:
        missing.append("LOCAL_LLM_API_KEY or LOCAL_LLM_API_KEY_FILE")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    return Settings(
        local_llm_base_url=os.environ["LOCAL_LLM_BASE_URL"].rstrip("/"),
        local_llm_api_key=api_key,
        local_llm_model=os.environ["LOCAL_LLM_MODEL"],
        local_llm_timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "600")),
        max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "5")),
        working_memory_max_messages=int(os.getenv("WORKING_MEMORY_MAX_MESSAGES", "12")),
        working_memory_max_chars=int(os.getenv("WORKING_MEMORY_MAX_CHARS", "12000")),
        summary_trigger_messages=int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "16")),
        memory_retrieval_limit=int(os.getenv("MEMORY_RETRIEVAL_LIMIT", "5")),
        memory_importance_threshold=float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "0.6")),
        planner_max_steps=int(os.getenv("PLANNER_MAX_STEPS", "8")),
        planner_max_replans=int(os.getenv("PLANNER_MAX_REPLANS", "2")),
        tool_max_retries=int(os.getenv("TOOL_MAX_RETRIES", "1")),
        confirm_write_actions=os.getenv("CONFIRM_WRITE_ACTIONS", "true").lower() == "true",
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
    )


def _load_api_key() -> str:
    inline_key = os.getenv("LOCAL_LLM_API_KEY", "").strip()
    if inline_key:
        return inline_key

    key_file = os.getenv("LOCAL_LLM_API_KEY_FILE", "").strip()
    if not key_file:
        return ""
    path = Path(key_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    key = path.read_text(encoding="utf-8").strip()
    if not key or key == "paste-your-deepseek-api-key-here":
        return ""
    return key
