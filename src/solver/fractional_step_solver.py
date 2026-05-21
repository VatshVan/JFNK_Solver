"""Fractional-step PETSc solver for the 2D lid-driven cavity with FGMRES momentum solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

try:
    from petsc4py import PETSc
except ImportError:  # PETSC_STUB
    PETSc = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from petsc4py import PETSc as PETScTyping

from src.bridge.petsc_torch_bridge import enforce_bcs
from src.telemetry.solver_logger import SolverLogger
from src.telemetry.timers import PETScTimer


ResidualFunction = Callable[["PETScTyping.Vec", "PETScTyping.Vec", float, float], None]


@dataclass
class SolveSummary:
    """Small telemetry carrier for solver smoke tests and logging hooks."""

    converged: bool
    reason: int
    iterations: int
    residual_norm: float


class FractionalStepSolver:
    """Advance an incompressible velocity field with predictor/projection splitting."""

    def __init__(
        self,
        da: "PETScTyping.DMDA",
        residual_fn: ResidualFunction | None = None,
        bc_type: str = "lid_driven_cavity",
        Re: float = 100.0,
        dt: float = 0.01,
        logger: SolverLogger | None = None,
        timestep: int = 0,
        monitor_ksp: bool = False,
    ) -> None:
        if PETSc is None:  # pragma: no cover - exercised only when PETSc is absent
            raise RuntimeError("petsc4py is required for FractionalStepSolver.")

        self.da = da
        self.residual_fn = residual_fn
        self.bc_type = bc_type
        self.Re = float(Re)
        self.dt = float(dt)
        self.logger = logger
        self.timestep = timestep
        self.monitor_ksp = monitor_ksp
        self.drrn_pc = None
        self.comm = da.getComm()
        nx, ny = (int(value) for value in da.getSizes()[:2])
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / max(nx - 1, 1)
        self.dy = 1.0 / max(ny - 1, 1)
        self.velocity_size = nx * ny * int(da.getDof())
        self.pressure_size = nx * ny
        self.pressure_da = PETSc.DMDA().create([nx, ny], dof=1, stencil_width=1, comm=self.comm)
        self.interior_nx = max(nx - 2, 0)
        self.interior_ny = max(ny - 2, 0)
        self.interior_pressure_size = self.interior_nx * self.interior_ny
        self._pressure_matrix = self._build_pressure_matrix()
        self._momentum_matrix: "PETScTyping.Mat | None" = None
        self._momentum_signature: tuple[float, float] | None = None
        self._pressure = self.pressure_da.createGlobalVec()
        self.last_predictor_summary: SolveSummary | None = None
        self.last_projection_summary: SolveSummary | None = None

    def predictor_step(self, u_n: "PETScTyping.Vec", dt: float, Re: float, max_it: int = 500) -> "PETScTyping.Vec":
        """Solve the semi-implicit momentum predictor for the cavity flow."""

        self.dt = float(dt)
        self.Re = float(Re)
        velocity = u_n.copy()
        enforce_bcs(velocity, self.da, self.bc_type)
        convection_u, convection_v = self._compute_convection(velocity)
        operator = self._get_momentum_matrix(dt=self.dt, reynolds=self.Re)
        rhs = self.da.createGlobalVec()
        rhs_array = self.da.getVecArray(rhs)
        velocity_array = self.da.getVecArray(velocity)
        rhs_array[1:-1, 1:-1, 0] = velocity_array[1:-1, 1:-1, 0] / self.dt - convection_u[1:-1, 1:-1]
        rhs_array[1:-1, 1:-1, 1] = velocity_array[1:-1, 1:-1, 1] / self.dt - convection_v[1:-1, 1:-1]
        enforce_bcs(rhs, self.da, self.bc_type)

        u_star = velocity.copy()
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(operator)

        # ── DRRN PRECONDITIONER HOOK ──────────────────────────────────────────────────
        # To enable DRRN-based preconditioning in Phase 3, replace pc_type='none' with
        # a custom PC shell:
        #     pc.setType(PETSc.PC.Type.SHELL)
        #     pc.setContext(drrn_preconditioner_object)
        #     pc.setApply(drrn_apply_fn)
        # The drrn_apply_fn receives the KSP residual Vec and writes the preconditioned
        # Vec in-place, calling vec_to_tensor → DRRN forward → tensor_to_vec internally.
        # ─────────────────────────────────────────────────────────────────────────────
        opts = PETSc.Options()
        opts["ksp_type"] = "fgmres"
        opts["ksp_gmres_restart"] = 50
        opts["pc_type"] = "none"
        opts["ksp_monitor_true_residual"] = None

        ksp.setType("fgmres")
        ksp.setTolerances(rtol=1e-8, max_it=max_it)
        ksp.setInitialGuessNonzero(True)
        ksp.setFromOptions()
        if self.drrn_pc is not None:
            from src.ml.drrn_preconditioner import attach_drrn_pc

            attach_drrn_pc(ksp, self.drrn_pc)
        if not self.monitor_ksp:
            ksp.cancelMonitor()
        with PETScTimer() as timer:
            ksp.solve(rhs, u_star)
        enforce_bcs(u_star, self.da, self.bc_type)

        self.last_predictor_summary = SolveSummary(
            converged=ksp.getConvergedReason() > 0,
            reason=int(ksp.getConvergedReason()),
            iterations=int(ksp.getIterationNumber()),
            residual_norm=float(ksp.getResidualNorm()),
        )
        self._log_predictor_metrics(ksp=ksp, velocity=u_star, elapsed_ms=timer.elapsed_ms)
        return u_star

    def projection_step(
        self,
        u_star: "PETScTyping.Vec",
        da: "PETScTyping.DMDA",
        dt: float = 1.0,
    ) -> tuple["PETScTyping.Vec", "PETScTyping.Vec"]:
        """Solve the PPE and apply the pressure correction."""

        self.dt = float(dt)
        tentative = u_star.copy()
        enforce_bcs(tentative, self.da, self.bc_type)
        rhs = PETSc.Vec().createSeq(self.interior_pressure_size, comm=self.comm)
        rhs_array = rhs.getArray()
        divergence = self.compute_divergence(tentative)
        rhs_array[:] = divergence[1:-1, 1:-1].reshape(-1) / max(self.dt, 1e-12)

        pressure_interior = rhs.duplicate()
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self._pressure_matrix)
        ksp.setType("cg")
        ksp.setTolerances(rtol=1e-10, max_it=1000)
        pc = ksp.getPC()
        pc.setType("icc")
        with PETScTimer() as timer:
            ksp.solve(rhs, pressure_interior)

        u_next = tentative.copy()
        u_array = self.da.getVecArray(u_next)
        pressure = self._pressure.copy()
        p_array = self.pressure_da.getVecArray(pressure)
        p_array[:, :] = 0.0
        p_array[1:-1, 1:-1] = pressure_interior.getArray().reshape(self.interior_ny, self.interior_nx)

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                px = p_array[j, i]
                py = p_array[j, i]
                if i < self.nx - 2:
                    px -= p_array[j, i + 1]
                if j < self.ny - 2:
                    py -= p_array[j + 1, i]
                u_array[j, i, 0] -= self.dt * px / self.dx
                u_array[j, i, 1] -= self.dt * py / self.dy
        enforce_bcs(u_next, self.da, self.bc_type)

        self.last_projection_summary = SolveSummary(
            converged=ksp.getConvergedReason() > 0,
            reason=int(ksp.getConvergedReason()),
            iterations=int(ksp.getIterationNumber()),
            residual_norm=float(ksp.getResidualNorm()),
        )
        self._pressure = pressure.copy()
        self._log_projection_metrics(ksp=ksp, velocity=u_next, pressure=pressure, elapsed_ms=timer.elapsed_ms)
        return u_next, pressure

    def compute_divergence(self, velocity: "PETScTyping.Vec") -> np.ndarray:
        """Return the cell-centered discrete divergence of a velocity vector."""

        array_view = self.da.getVecArray(velocity)
        divergence = np.zeros((self.ny, self.nx), dtype=np.float64)
        divergence[1:-1, 1:-1] = (
            (array_view[1:-1, 1:-1, 0] - array_view[1:-1, :-2, 0]) / self.dx
            + (array_view[1:-1, 1:-1, 1] - array_view[:-2, 1:-1, 1]) / self.dy
        )
        return divergence

    def divergence_norms(self, velocity: "PETScTyping.Vec") -> tuple[float, float]:
        """Return L2 and Linf norms for the discrete divergence field."""

        divergence = self.compute_divergence(velocity)
        return float(np.linalg.norm(divergence.ravel())), float(np.max(np.abs(divergence)))

    def _build_velocity_matrix(self, dt: float, reynolds: float) -> "PETScTyping.Mat":
        matrix = PETSc.Mat().createAIJ([self.velocity_size, self.velocity_size], comm=self.comm)
        matrix.setUp()
        viscosity = 1.0 / max(reynolds, 1e-12)
        scale_x = viscosity / (self.dx * self.dx)
        scale_y = viscosity / (self.dy * self.dy)

        for j in range(self.ny):
            for i in range(self.nx):
                boundary = i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1
                for comp in range(2):
                    row = self._velocity_index(i, j, comp)
                    if boundary:
                        matrix.setValue(row, row, 1.0)
                        continue

                    diag = 1.0 / dt + 2.0 * scale_x + 2.0 * scale_y
                    matrix.setValue(row, row, diag)
                    matrix.setValue(row, self._velocity_index(i - 1, j, comp), -scale_x)
                    matrix.setValue(row, self._velocity_index(i + 1, j, comp), -scale_x)
                    matrix.setValue(row, self._velocity_index(i, j - 1, comp), -scale_y)
                    matrix.setValue(row, self._velocity_index(i, j + 1, comp), -scale_y)

        matrix.assemblyBegin()
        matrix.assemblyEnd()
        return matrix

    def _build_pressure_matrix(self) -> "PETScTyping.Mat":
        matrix = PETSc.Mat().createAIJ([self.pressure_size, self.pressure_size], comm=self.comm)
        matrix.destroy()
        matrix = PETSc.Mat().createAIJ([self.interior_pressure_size, self.interior_pressure_size], comm=self.comm)
        matrix.setUp()
        scale_x = 1.0 / (self.dx * self.dx)
        scale_y = 1.0 / (self.dy * self.dy)

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                row = self._interior_pressure_index(i, j)
                diag = scale_x + scale_y
                if i > 1:
                    matrix.setValue(row, self._interior_pressure_index(i - 1, j), -scale_x)
                    diag += scale_x
                if i < self.nx - 2:
                    matrix.setValue(row, self._interior_pressure_index(i + 1, j), -scale_x)
                if j > 1:
                    matrix.setValue(row, self._interior_pressure_index(i, j - 1), -scale_y)
                    diag += scale_y
                if j < self.ny - 2:
                    matrix.setValue(row, self._interior_pressure_index(i, j + 1), -scale_y)
                matrix.setValue(row, row, diag)

        matrix.assemblyBegin()
        matrix.assemblyEnd()
        matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        matrix.setOption(PETSc.Mat.Option.SPD, True)
        symmetric_type = "seqsbaij" if self.comm.getSize() == 1 else "mpisbaij"
        matrix = matrix.convert(symmetric_type)
        matrix.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        matrix.setOption(PETSc.Mat.Option.SPD, True)
        return matrix

    def _compute_convection(self, velocity: "PETScTyping.Vec") -> tuple[np.ndarray, np.ndarray]:
        array_view = self.da.getVecArray(velocity)
        u = array_view[:, :, 0]
        v = array_view[:, :, 1]
        conv_u = np.zeros((self.ny, self.nx), dtype=np.float64)
        conv_v = np.zeros((self.ny, self.nx), dtype=np.float64)

        du_dx = np.zeros_like(conv_u)
        du_dy = np.zeros_like(conv_u)
        dv_dx = np.zeros_like(conv_u)
        dv_dy = np.zeros_like(conv_u)
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * self.dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * self.dy)
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * self.dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * self.dy)
        conv_u[1:-1, 1:-1] = u[1:-1, 1:-1] * du_dx[1:-1, 1:-1] + v[1:-1, 1:-1] * du_dy[1:-1, 1:-1]
        conv_v[1:-1, 1:-1] = u[1:-1, 1:-1] * dv_dx[1:-1, 1:-1] + v[1:-1, 1:-1] * dv_dy[1:-1, 1:-1]
        return conv_u, conv_v

    def _get_momentum_matrix(self, dt: float, reynolds: float) -> "PETScTyping.Mat":
        signature = (float(dt), float(reynolds))
        if self._momentum_matrix is None or self._momentum_signature != signature:
            self._momentum_matrix = self._build_velocity_matrix(dt=dt, reynolds=reynolds)
            self._momentum_signature = signature
        return self._momentum_matrix

    def _compute_cfl(self, velocity: "PETScTyping.Vec") -> float:
        array_view = self.da.getVecArray(velocity)
        u = np.abs(array_view[:, :, 0]).max()
        v = np.abs(array_view[:, :, 1]).max()
        return float(max(u * self.dt / max(self.dx, 1e-12), v * self.dt / max(self.dy, 1e-12)))

    def _safe_condition_estimate(self, ksp: "PETScTyping.KSP") -> tuple[float, float, float]:
        try:
            min_sv, max_sv = ksp.computeExtremeSingularValues()
            min_sv = float(min_sv)
            max_sv = float(max_sv)
            if min_sv <= 0.0:
                return min_sv, max_sv, float("inf")
            return min_sv, max_sv, max_sv / min_sv
        except Exception:
            return 0.0, 0.0, 0.0

    def _log_predictor_metrics(self, ksp: "PETScTyping.KSP", velocity: "PETScTyping.Vec", elapsed_ms: float) -> None:
        if self.logger is None:
            return
        min_sv, max_sv, cond = self._safe_condition_estimate(ksp)
        div_l2, div_linf = self.divergence_norms(velocity)
        self.logger.log_metrics(
            {
                "timestep": self.timestep,
                "Re": self.Re,
                "fgmres_iters": int(ksp.getIterationNumber()),
                "cond_min_sv": min_sv,
                "cond_max_sv": max_sv,
                "cond_estimate": cond,
                "div_l2": div_l2,
                "div_linf": div_linf,
                "cfl_max": self._compute_cfl(velocity),
                "t_petsc_solve_ms": elapsed_ms,
            }
        )

    def _log_projection_metrics(
        self,
        ksp: "PETScTyping.KSP",
        velocity: "PETScTyping.Vec",
        pressure: "PETScTyping.Vec",
        elapsed_ms: float,
    ) -> None:
        if self.logger is None:
            return
        div_l2, div_linf = self.divergence_norms(velocity)
        self.logger.log_metrics(
            {
                "timestep": self.timestep,
                "Re": self.Re,
                "ppe_iters": int(ksp.getIterationNumber()),
                "div_l2": div_l2,
                "div_linf": div_linf,
                "pressure_norm": float(pressure.norm()),
                "t_ppe_solve_ms": elapsed_ms,
            }
        )

    def _velocity_index(self, i: int, j: int, comp: int) -> int:
        return ((j * self.nx) + i) * 2 + comp

    def _pressure_index(self, i: int, j: int) -> int:
        return (j * self.nx) + i

    def _interior_pressure_index(self, i: int, j: int) -> int:
        return ((j - 1) * self.interior_nx) + (i - 1)
