"""Run Stage 0 CFD generation, Stage 1 warm-up, and Stage 2 continuation for the DRRN pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bridge.petsc_torch_bridge import enforce_bcs
from src.ml.drrn import DRRN
from src.ml.drrn_config import DRRNConfig
from src.ml.physics_loss import divergence_loss, physics_informed_loss
from src.solver.fractional_step_solver import FractionalStepSolver
from src.telemetry.solver_logger import SolverLogger
from src.telemetry.timers import CUDATimer

try:
    from petsc4py import PETSc
except ImportError:  # PETSC_STUB
    PETSc = None  # type: ignore[assignment]

try:
    from scipy.ndimage import zoom as scipy_zoom
except ImportError:  # pragma: no cover - exercised only when SciPy is unavailable
    scipy_zoom = None


class FlowFieldDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load one coarse/fine steady-state cavity field pair from an HDF5 file."""

    def __init__(self, h5_path: str | Path, coarse_res: int, fine_res: int) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as handle:
            coarse_key = self._field_key(handle, "u", coarse_res)
            fine_key = self._field_key(handle, "u", fine_res)
            self.coarse = torch.from_numpy(
                np.stack((handle[coarse_key][:], handle[self._field_key(handle, "v", coarse_res)][:]), axis=0)
            ).float()
            self.fine = torch.from_numpy(
                np.stack((handle[fine_key][:], handle[self._field_key(handle, "v", fine_res)][:]), axis=0)
            ).float()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coarse, self.fine

    @staticmethod
    def _field_key(handle: h5py.File, prefix: str, resolution: int) -> str:
        preferred = f"{prefix}_{resolution}"
        if preferred in handle:
            return preferred
        if resolution == 64 and f"{prefix}_200" in handle and handle[f"{prefix}_200"].shape == (64, 64):
            return f"{prefix}_200"
        raise KeyError(f"Dataset {preferred!r} not found in {handle.filename!r}.")


def _parse_re_values(values: Iterable[float] | None, config: DRRNConfig) -> list[float]:
    if values:
        return [float(value) for value in values]
    return [float(value) for value in np.arange(config.Re_start, config.Re_max + config.Re_step, config.Re_step)]


def _device_for_run(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resize_bilinear(array: np.ndarray, size: int) -> np.ndarray:
    if scipy_zoom is not None:
        zoom_factor = size / array.shape[-1]
        return scipy_zoom(array, zoom=zoom_factor, order=1)

    tensor = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def _make_re_solver(
    reynolds: float,
    grid_size: int,
    dt: float,
    logger: SolverLogger | None = None,
) -> tuple["PETSc.DMDA", FractionalStepSolver]:
    if PETSc is None:  # pragma: no cover - exercised only when PETSc is unavailable
        raise RuntimeError("petsc4py is required for Stage 0/2 solver execution.")
    da = PETSc.DMDA().create([grid_size, grid_size], dof=2, stencil_width=1, comm=PETSc.COMM_SELF)
    solver = FractionalStepSolver(da, Re=reynolds, dt=dt, logger=logger)
    return da, solver


def generate_stage0_data(
    config: DRRNConfig,
    data_dir: Path,
    re_values: list[float],
    logger: SolverLogger,
) -> list[dict[str, float]]:
    """Generate steady lid-driven cavity fields and downsampled HDF5 training data."""

    data_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, float]] = []
    fine_size = int(config.fine_grid_size)
    coarse_sizes = [2, 5, 10, 16, 25, 32, 50]

    u = None
    pressure = None

    for reynolds in re_values:
        da, solver = _make_re_solver(reynolds=reynolds, grid_size=fine_size, dt=config.dt, logger=logger)
        if u is None:
            u = da.createGlobalVec()
            enforce_bcs(u, da, solver.bc_type)
        else:
            enforce_bcs(u, da, solver.bc_type)
            
        if pressure is None:
            pressure = solver.pressure_da.createGlobalVec()
            
        converged_step = config.max_steps
        rel_change = float("inf")
        div_l2 = float("inf")
        div_linf = float("inf")

        for step in range(1, config.max_steps + 1):
            solver.timestep = step
            previous = u.getArray(readonly=True).copy()
            u_star = solver.predictor_step(u, config.dt, reynolds)
            u, pressure = solver.projection_step(u_star, da, config.dt)
            current = u.getArray(readonly=True)
            rel_change = float(np.linalg.norm(current - previous) / (np.linalg.norm(previous) + 1e-14))
            div_l2, div_linf = solver.divergence_norms(u)
            if rel_change < config.steady_tol:
                converged_step = step
                break

        velocity_array = da.getVecArray(u)
        pressure_array = solver.pressure_da.getVecArray(pressure)
        u_fine = velocity_array[:, :, 0].copy()
        v_fine = velocity_array[:, :, 1].copy()
        p_fine = pressure_array[:, :].copy()

        output_path = data_dir / f"Re_{reynolds:05.0f}.h5"
        with h5py.File(output_path, "w") as handle:
            # Dynamic grid size support for 128x128 grid
            handle[f"u_{fine_size}"] = u_fine
            handle[f"v_{fine_size}"] = v_fine
            handle[f"p_{fine_size}"] = p_fine
            # Backwards compatibility keys
            handle["u_64"] = u_fine
            handle["v_64"] = v_fine
            handle["p_64"] = p_fine
            handle["u_200"] = u_fine
            handle["v_200"] = v_fine
            handle["p_200"] = p_fine
            for coarse_size in coarse_sizes:
                handle[f"u_{coarse_size}"] = _resize_bilinear(u_fine, coarse_size)
                handle[f"v_{coarse_size}"] = _resize_bilinear(v_fine, coarse_size)

        summaries.append(
            {
                "Re": reynolds,
                "converged_step": float(converged_step),
                "rel_change": rel_change,
                "div_l2": div_l2,
                "div_linf": div_linf,
            }
        )
        logger.log_message(
            f"Stage0 Re={reynolds:.0f} converged_step={converged_step} rel_change={rel_change:.3e} div_l2={div_l2:.3e}"
        )

    return summaries


def _build_warmup_dataset(config: DRRNConfig, data_dir: Path, re_values: list[float]) -> ConcatDataset:
    datasets = [
        FlowFieldDataset(data_dir / f"Re_{reynolds:05.0f}.h5", coarse_res=config.coarse_res, fine_res=config.fine_res)
        for reynolds in re_values
    ]
    return ConcatDataset(datasets)


def _make_loaders(dataset: Dataset[tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> tuple[DataLoader[Any], DataLoader[Any]]:
    val_size = max(1, len(dataset) // 3)
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(7))
    # Sandbox restriction: use num_workers=0 and pin_memory=False for local CPU-safe execution.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader


def _save_checkpoint(model: nn.Module, checkpoint_dir: Path, tag: str, epoch: int) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{tag}_epoch{epoch:04d}.pt"
    torch.save(model.state_dict(), path)
    return path


def _apply_tensor_cavity_bcs(field: torch.Tensor) -> torch.Tensor:
    """Clamp DRRN outputs to the lid-driven cavity wall values before scoring."""

    field = field.clone()
    field[:, :, 0, :] = 0.0
    field[:, :, :, 0] = 0.0
    field[:, :, :, -1] = 0.0
    field[:, 0, -1, :] = 1.0
    field[:, 1, -1, :] = 0.0
    return field


def _train_model(
    model: DRRN,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: DRRNConfig,
    epochs: int,
    device: torch.device,
    checkpoint_dir: Path,
    checkpoint_tag: str,
    is_warmstart: bool = False,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    
    if is_warmstart:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=30,
            min_lr=1e-6,
            verbose=True
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
        
    dx = dy = 1.0 / max(config.fine_res - 1, 1)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_div = 0.0
        train_batches = 0

        for coarse, fine in train_loader:
            coarse = coarse.to(device=device, dtype=torch.float32)
            fine = fine.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = _apply_tensor_cavity_bcs(model(coarse))
            loss, recon, div = physics_informed_loss(
                u_fine_pred=pred,
                u_fine_true=fine,
                lambda_div=config.lambda_div,
                dx=dx,
                dy=dy,
                padding_mode=config.padding_mode,
            )
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach().cpu())
            train_recon += float(recon.detach().cpu())
            train_div += float(div.detach().cpu())
            train_batches += 1

        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_div = 0.0
        val_batches = 0
        with torch.no_grad():
            for coarse, fine in val_loader:
                coarse = coarse.to(device=device, dtype=torch.float32)
                fine = fine.to(device=device, dtype=torch.float32)
                pred = _apply_tensor_cavity_bcs(model(coarse))
                loss, recon, div = physics_informed_loss(
                    u_fine_pred=pred,
                    u_fine_true=fine,
                    lambda_div=config.lambda_div,
                    dx=dx,
                    dy=dy,
                    padding_mode=config.padding_mode,
                )
                val_total += float(loss.detach().cpu())
                val_recon += float(recon.detach().cpu())
                val_div += float(div.detach().cpu())
                val_batches += 1

        val_div_avg = val_div / max(val_batches, 1)
        if is_warmstart:
            scheduler.step(val_div_avg)
        else:
            scheduler.step()

        record = {
            "epoch": float(epoch),
            "train_loss": train_total / max(train_batches, 1),
            "train_recon": train_recon / max(train_batches, 1),
            "train_divergence": train_div / max(train_batches, 1),
            "val_loss": val_total / max(val_batches, 1),
            "val_recon": val_recon / max(val_batches, 1),
            "val_divergence": val_div_avg,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)

        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:4d}  val_loss={record['val_loss']:.6f}  val_recon={record['val_recon']:.6f}  val_div={record['val_divergence']:.6f}  lr={record['lr']:.2e}", flush=True)

        if epoch % config.checkpoint_every == 0 or epoch == epochs:
            _save_checkpoint(model, checkpoint_dir=checkpoint_dir, tag=checkpoint_tag, epoch=epoch)

        if is_warmstart and val_div_avg < 0.05:
            print(f"Early stopping at epoch {epoch} because divergence norm {val_div_avg:.6f} < 0.05", flush=True)
            _save_checkpoint(model, checkpoint_dir=checkpoint_dir, tag=checkpoint_tag, epoch=epoch)
            break

    final_divergence = history[-1]["val_divergence"]
    if is_warmstart:
        assert final_divergence < 0.1, (
            f"Warm-up gate failed: div_norm={final_divergence:.4f}. "
            f"Try: increase lambda_div further, or run 200 more epochs."
        )

    summary = {
        "final_loss": history[-1]["val_recon"],
        "final_objective": history[-1]["val_loss"],
        "final_divergence": final_divergence,
        "epochs": float(len(history)),
    }
    return history, summary


def run_warmup(
    config: DRRNConfig,
    *,
    epochs: int,
    device: torch.device,
    data_dir: Path,
    checkpoint_dir: Path,
    log_path: Path,
    re_values: list[float],
    warmstart_checkpoint: Path | None = None,
) -> dict[str, float]:
    """Train the coarse-to-fine DRRN on real Stage 0 cavity data."""

    dataset = _build_warmup_dataset(config=config, data_dir=data_dir, re_values=re_values)
    train_loader, val_loader = _make_loaders(dataset, batch_size=config.batch_size)
    model = DRRN(
        n_resblocks=config.n_resblocks,
        n_feats=config.n_feats,
        scale=config.scale,
        res_scale=config.res_scale,
    ).to(device=device, dtype=torch.float32)
    
    if warmstart_checkpoint is not None:
        state_dict = torch.load(warmstart_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded warmstart checkpoint: {warmstart_checkpoint}", flush=True)
        
    model.count_parameters()

    history, summary = _train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        epochs=epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        checkpoint_tag="drrn_warmup_finetuned" if warmstart_checkpoint else "drrn_warmup",
        is_warmstart=(warmstart_checkpoint is not None),
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record) + "\n")

    summary["checkpoint"] = str(checkpoint_dir / f"{'drrn_warmup_finetuned' if warmstart_checkpoint else 'drrn_warmup'}_epoch{int(summary['epochs']):04d}.pt")
    return summary


def _load_model_from_checkpoint(config: DRRNConfig, checkpoint: Path, device: torch.device) -> DRRN:
    model = DRRN(
        n_resblocks=config.n_resblocks,
        n_feats=config.n_feats,
        scale=config.scale,
        res_scale=config.res_scale,
    ).to(device=device, dtype=torch.float32)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_matching_state(target: nn.Module, source_state: dict[str, torch.Tensor]) -> None:
    """Load only the checkpoint tensors whose names and shapes match the target module."""

    target_state = target.state_dict()
    filtered = {name: tensor for name, tensor in source_state.items() if name in target_state and target_state[name].shape == tensor.shape}
    target.load_state_dict(filtered, strict=False)


def _write_prediction_to_vec(tensor: torch.Tensor, da: "PETSc.DMDA") -> "PETSc.Vec":
    vec = da.createGlobalVec()
    array = da.getVecArray(vec)
    prediction = tensor.squeeze(0).detach().cpu().numpy()
    array[:, :, 0] = prediction[0]
    array[:, :, 1] = prediction[1]
    enforce_bcs(vec, da, "lid_driven_cavity")
    return vec


def drrn_two_phase_solve(solver, u_init, dt, Re, model, da, coarse_res, n_inner=5):
    """
    Phase 1: Run FGMRES for n_inner iterations from u_init.
             Use solver.predictor_step() but limit FGMRES to n_inner iterations:
             set ksp.setTolerances(max_it=n_inner) for this call only.
    Phase 2: Apply DRRN to the Phase 1 output to smooth it.
             This uses vec_to_tensor → DRRN forward → tensor_to_vec.
             Then enforce_bcs() to restore wall BCs.
    Phase 3: Run FGMRES to full convergence from the DRRN-smoothed solution.
    Returns: (u_final, total_inner_iters_phase1 + total_inner_iters_phase3)
    """
    # Phase 1: limit FGMRES iterations
    u_phase1 = solver.predictor_step(u_init, dt, Re, max_it=n_inner)
    iters_phase1 = float(solver.last_predictor_summary.iterations)
    
    # Phase 2: Downsample to coarse resolution and apply DRRN
    u_phase1_arr = da.getVecArray(u_phase1)
    u_np = u_phase1_arr[:, :, 0].copy()
    v_np = u_phase1_arr[:, :, 1].copy()
    
    u_coarse = _resize_bilinear(u_np, coarse_res)
    v_coarse = _resize_bilinear(v_np, coarse_res)
    
    model_device = next(model.parameters()).device
    coarse_tensor = torch.from_numpy(np.stack([u_coarse, v_coarse], axis=0)).float().unsqueeze(0).to(device=model_device)
    
    with torch.no_grad():
        prediction = _apply_tensor_cavity_bcs(model(coarse_tensor))
        
    u_smoothed = _write_prediction_to_vec(prediction, da)
    
    # Phase 3: Solve to full convergence from the DRRN-smoothed solution
    u_final = solver.predictor_step(u_smoothed, dt, Re, max_it=500)
    iters_phase3 = float(solver.last_predictor_summary.iterations)
    
    total_iters = iters_phase1 + iters_phase3
    return u_final, total_iters


def run_continuation(
    config: DRRNConfig,
    *,
    device: torch.device,
    data_dir: Path,
    checkpoint_dir: Path,
    logger: SolverLogger,
    re_values: list[float],
    warmup_checkpoint: Path | None,
    use_drrn_pc: bool = False,
    use_drrn_twophase: bool = False,
) -> list[dict[str, float]]:
    """Fine-tune the DRRN across Reynolds continuation steps and log iteration reductions."""

    if warmup_checkpoint is None:
        raise ValueError("A warm-up checkpoint is required before Stage 2 continuation.")

    model = _load_model_from_checkpoint(config=config, checkpoint=warmup_checkpoint, device=device)
    results: list[dict[str, float]] = []
    continuation_values = [value for value in re_values if value >= config.Re_start]

    for continuation_step, reynolds in enumerate(continuation_values, start=1):
        dataset = FlowFieldDataset(data_dir / f"Re_{reynolds:05.0f}.h5", config.coarse_res, config.fine_res)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        history, _ = _train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            epochs=config.finetune_epochs,
            device=device,
            checkpoint_dir=checkpoint_dir,
            checkpoint_tag=f"drrn_Re{reynolds:05.0f}_finetune",
        )

        da, solver_baseline = _make_re_solver(reynolds=reynolds, grid_size=config.fine_grid_size, dt=config.dt, logger=logger)
        u_zero = da.createGlobalVec()
        enforce_bcs(u_zero, da, solver_baseline.bc_type)
        solver_baseline.timestep = continuation_step
        solver_baseline.predictor_step(u_zero, config.dt, reynolds)
        baseline_iters = float(solver_baseline.last_predictor_summary.iterations)

        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with h5py.File(data_dir / f"Re_{reynolds:05.0f}.h5", "r") as handle:
            coarse = torch.from_numpy(
                np.stack((handle[f"u_{config.coarse_res}"][:], handle[f"v_{config.coarse_res}"][:]), axis=0)
            ).float()

        inference_device = device if device.type == "cuda" else torch.device("cpu")
        model.to(device=inference_device, dtype=torch.float32)
        coarse = coarse.unsqueeze(0).to(device=inference_device, dtype=torch.float32)
        if inference_device.type == "cuda":
            with CUDATimer() as timer:
                with torch.no_grad():
                    prediction = _apply_tensor_cavity_bcs(model(coarse))
            inference_ms = timer.elapsed_ms
        else:
            start = time.perf_counter()
            with torch.no_grad():
                prediction = _apply_tensor_cavity_bcs(model(coarse))
            inference_ms = (time.perf_counter() - start) * 1_000.0

        prediction_vec = _write_prediction_to_vec(prediction, da)
        drrn_div_l2, drrn_div_linf = solver_baseline.divergence_norms(prediction_vec)

        if use_drrn_twophase:
            da_drrn, solver_drrn = _make_re_solver(reynolds=reynolds, grid_size=config.fine_grid_size, dt=config.dt, logger=logger)
            solver_drrn.timestep = continuation_step
            u_init = da_drrn.createGlobalVec()
            enforce_bcs(u_init, da_drrn, solver_drrn.bc_type)
            u_final, drrn_iters = drrn_two_phase_solve(
                solver=solver_drrn,
                u_init=u_init,
                dt=config.dt,
                Re=reynolds,
                model=model,
                da=da_drrn,
                coarse_res=config.coarse_res,
                n_inner=5
            )
        else:
            da_drrn, solver_drrn = _make_re_solver(reynolds=reynolds, grid_size=config.fine_grid_size, dt=config.dt, logger=logger)
            solver_drrn.timestep = continuation_step
            if use_drrn_pc:
                from src.ml.drrn_preconditioner import DRRNPreconditioner

                model_cpu = DRRN(
                    n_resblocks=config.n_resblocks,
                    n_feats=config.n_feats,
                    scale=1,
                    res_scale=config.res_scale,
                )
                _load_matching_state(model_cpu, model.state_dict())
                solver_drrn.drrn_pc = DRRNPreconditioner(model_cpu.eval(), da_drrn, device="cpu")
            solver_drrn.predictor_step(prediction_vec, config.dt, reynolds)
            drrn_iters = float(solver_drrn.last_predictor_summary.iterations)

        acceleration = baseline_iters / max(drrn_iters, 1.0)
        record = {
            "Re": reynolds,
            "timestep": continuation_step,
            "fgmres_baseline": baseline_iters,
            "fgmres_drrn": drrn_iters,
            "acceleration_factor": acceleration,
            "t_drrn_inference_ms": inference_ms,
            "drrn_div_norm": drrn_div_l2,
            "div_l2": drrn_div_l2,
            "div_linf": drrn_div_linf,
            "cond_estimate": solver_drrn.last_predictor_summary.residual_norm,
            "t_petsc_solve_ms": 0.0,
            "pc_type": "drrn_twophase" if use_drrn_twophase else ("drrn_shell" if use_drrn_pc else "none"),
            "fgmres_iters": drrn_iters,
        }
        logger.log_metrics(record)
        print(
            f"Re={reynolds:.0f}  baseline={int(baseline_iters)}  drrn={int(drrn_iters)}  speedup={acceleration:.2f}x"
        )
        results.append(
            {
                "Re": reynolds,
                "baseline_iters": baseline_iters,
                "drrn_iters": drrn_iters,
                "acceleration_factor": acceleration,
                "drrn_div_norm": drrn_div_l2,
                "finetune_final_loss": history[-1]["val_loss"],
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the DRRN continuation pipeline.")
    parser.add_argument("--stage", choices=["stage0", "warmup", "continuation", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for the warm-up stage.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fields"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--warmup-log", type=Path, default=Path("logs/drrn_warmup.jsonl"))
    parser.add_argument("--re-values", type=float, nargs="*", default=[100.0, 200.0, 300.0])
    parser.add_argument("--use-drrn-pc", action="store_true")
    parser.add_argument("--use-drrn-twophase", action="store_true")
    parser.add_argument("--warmup-checkpoint", type=Path, default=None)
    parser.add_argument("--lambda-div", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DRRNConfig()
    if args.lambda_div is not None:
        config = replace(config, lambda_div=float(args.lambda_div))
    if args.lr is not None:
        config = replace(config, lr=float(args.lr))
    if args.dt is not None:
        config = replace(config, dt=float(args.dt))
    if args.max_steps is not None:
        config = replace(config, max_steps=int(args.max_steps))
    device = _device_for_run(args.device)
    re_values = _parse_re_values(args.re_values, config)
    logger = SolverLogger(log_dir="logs", logger_name="drrn_pipeline")

    print(json.dumps({"config": asdict(config), "device": str(device), "re_values": re_values}, indent=2))

    stage0_summary: list[dict[str, float]] = []
    warmup_summary: dict[str, float] | None = None
    continuation_summary: list[dict[str, float]] = []

    if args.stage in {"stage0", "full"}:
        stage0_summary = generate_stage0_data(config=config, data_dir=args.data_dir, re_values=re_values, logger=logger)
        print(json.dumps({"stage0": stage0_summary}, indent=2))

    if args.stage in {"warmup", "full"}:
        warmup_summary = run_warmup(
            config=config,
            epochs=args.epochs,
            device=device,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            log_path=args.warmup_log,
            re_values=re_values,
            warmstart_checkpoint=args.warmup_checkpoint,
        )
        print(json.dumps({"warmup": warmup_summary}, indent=2))

    if args.stage == "continuation" or args.stage == "full":
        warmup_checkpoint = args.warmup_checkpoint
        if warmup_checkpoint is None:
            if warmup_summary is None:
                raise ValueError("Provide --warmup-checkpoint when running continuation without warm-up.")
            warmup_checkpoint = Path(warmup_summary["checkpoint"])
        continuation_summary = run_continuation(
            config=config,
            device=device,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            logger=logger,
            re_values=re_values,
            warmup_checkpoint=warmup_checkpoint,
            use_drrn_pc=args.use_drrn_pc,
            use_drrn_twophase=args.use_drrn_twophase,
        )
        print(json.dumps({"continuation": continuation_summary}, indent=2))


if __name__ == "__main__":
    main()
