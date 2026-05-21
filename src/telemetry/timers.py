"""Timing helpers for PETSc wall-clock segments and synchronized CUDA inference blocks."""

from __future__ import annotations

import time

import torch

try:
    import mpi4py

    mpi4py.rc.initialize = False
    from mpi4py import MPI
except ImportError:  # pragma: no cover - exercised only when mpi4py is unavailable
    MPI = None


class PETScTimer:
    """Context manager that prefers MPI wall time and falls back to perf_counter."""

    def __init__(self) -> None:
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> "PETScTimer":
        self._start = MPI.Wtime() if MPI is not None else time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        end = MPI.Wtime() if MPI is not None else time.perf_counter()
        self.elapsed_ms = (end - self._start) * 1_000.0


class CUDATimer:
    """Context manager for synchronized CUDA timing with CPU fallback semantics."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = torch.cuda.is_available() if enabled is None else enabled
        self.elapsed_ms = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._cpu_start = 0.0

    def __enter__(self) -> "CUDATimer":
        if self.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self.enabled and torch.cuda.is_available():
            assert self._start_event is not None
            assert self._end_event is not None
            self._end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = float(self._start_event.elapsed_time(self._end_event))
        else:
            self.elapsed_ms = (time.perf_counter() - self._cpu_start) * 1_000.0
