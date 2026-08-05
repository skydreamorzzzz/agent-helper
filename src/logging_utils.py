from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def setup_run_logger(run_id: str, logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"agent.run.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.FileHandler(logs_dir / f"{run_id}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **data: Any) -> None:
    payload = {"event": event, **data}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))

