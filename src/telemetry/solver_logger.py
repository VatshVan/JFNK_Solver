"""Structured telemetry logging for PETSc solves, DRRN inference, and stability monitors."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive conversion fallback
            return value
    return value


class SolverLogger:
    """Write human-readable and machine-readable telemetry streams side by side."""

    def __init__(
        self,
        log_dir: str | Path = "logs",
        text_filename: str = "solver.log",
        metrics_filename: str = "metrics.jsonl",
        logger_name: str = "jfnk_solver",
        max_bytes: int = 1_000_000,
        backup_count: int = 3,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.log_dir / text_filename
        self.metrics_path = self.log_dir / metrics_filename

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()

        handler = RotatingFileHandler(
            self.text_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(handler)

    def log_message(self, message: str, level: int = logging.INFO) -> None:
        """Write a plain-text event to the rotating text log."""

        self.logger.log(level, message)

    def log_metrics(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Append one JSON-Lines telemetry record and mirror it to the text log."""

        record = {key: _to_jsonable(value) for key, value in metrics.items()}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.logger.info("metrics=%s", json.dumps(record, sort_keys=True))
        return record
