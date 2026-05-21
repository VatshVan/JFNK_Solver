"""Unit tests for telemetry logging and timing helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.telemetry.solver_logger import SolverLogger
from src.telemetry.timers import PETScTimer


def test_solver_logger_writes_valid_jsonl(tmp_path: Path) -> None:
    logger = SolverLogger(log_dir=tmp_path, logger_name="telemetry_case_one")
    record = logger.log_metrics({"timestep": 1, "fgmres_iters": 8, "Re": 100.0})
    lines = logger.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_solver_logger_appends_second_line(tmp_path: Path) -> None:
    logger = SolverLogger(log_dir=tmp_path, logger_name="telemetry_case_two")
    logger.log_metrics({"timestep": 1, "fgmres_iters": 8})
    logger.log_metrics({"timestep": 2, "fgmres_iters": 6})
    lines = logger.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_metrics_jsonl_is_pandas_readable(tmp_path: Path) -> None:
    logger = SolverLogger(log_dir=tmp_path, logger_name="telemetry_case_three")
    logger.log_metrics({"timestep": 1, "fgmres_iters": 8, "cond_estimate": 2.5})
    logger.log_metrics({"timestep": 2, "fgmres_iters": 5, "cond_estimate": 1.9})
    frame = pd.read_json(logger.metrics_path, lines=True)
    assert list(frame["timestep"]) == [1, 2]


def test_petsc_timer_reports_positive_elapsed_ms() -> None:
    with PETScTimer() as timer:
        time.sleep(0.01)
    assert timer.elapsed_ms > 0.0
