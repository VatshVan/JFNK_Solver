"""
Generates VTK/VTU output for the Re=500 lid-driven cavity.
Run 20 timesteps from rest and write the velocity, pressure, and DRRN
error field at each step to outputs/vtk/.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import glob
import numpy as np
import h5py
import torch

from petsc4py import PETSc
from src.solver.fractional_step_solver import FractionalStepSolver
from src.bridge.petsc_torch_bridge import enforce_bcs
from src.visualization.vtk_writer import write_vtk
from src.ml.drrn import DRRN
from src.ml.drrn_config import DRRNConfig

Re = 500.0
N = 128
dt = 0.005
config = DRRNConfig()

da = PETSc.DMDA().create([N, N], dof=2, stencil_width=1, comm=PETSc.COMM_SELF)
solver = FractionalStepSolver(da, Re=Re, dt=dt)
u = da.createGlobalVec()
enforce_bcs(u, da, solver.bc_type)

# ── Load DRRN ──────────────────────────────────────────────────────────────
model = DRRN(
    n_resblocks=config.n_resblocks,
    n_feats=config.n_feats,
    scale=config.scale,
)
ckpts = sorted(glob.glob("checkpoints/drrn_warmup_epoch*.pt"))
assert ckpts, "No checkpoint found — run Stage 1 (warmup) first"
print(f"Loading checkpoint: {ckpts[-1]}")
model.load_state_dict(torch.load(ckpts[-1], map_location="cpu"))
model.eval()

# ── Load coarse field for DRRN input ───────────────────────────────────────
with h5py.File(f"data/fields/Re_{Re:05.0f}.h5", "r") as f:
    u_c = torch.from_numpy(
        np.stack([f["u_32"][:], f["v_32"][:]], axis=0)
    ).float()

with torch.no_grad():
    u_drrn = model(u_c.unsqueeze(0)).squeeze(0).numpy()  # (2, 128, 128)

print(f"DRRN prediction shape: {u_drrn.shape}")
print(f"Running 20 timesteps at Re={Re:.0f}, N={N}x{N}, dt={dt}")

for step in range(20):
    u_star = solver.predictor_step(u, dt=dt, Re=Re)
    u, p = solver.projection_step(u_star, da, dt=dt)

    # ── Build scalar PETSc Vecs for velocity components ─────────────────
    # Velocity DA (dof=2) — we pass the full velocity Vec directly
    # and use a scalar pressure DA for pressure

    # Current solver velocity array (N, N, 2)
    u_arr = da.getVecArray(u)  # shape (N, N, 2)
    u_np = u_arr[:, :, 0].copy()
    v_np = u_arr[:, :, 1].copy()

    # Compute DRRN error field magnitude
    err_u = np.abs(u_drrn[0] - u_np)
    err_v = np.abs(u_drrn[1] - v_np)
    err_mag = np.sqrt(err_u**2 + err_v**2)

    # Build scalar DAs for single-component fields
    da_scalar = PETSc.DMDA().create([N, N], dof=1, stencil_width=1, comm=PETSc.COMM_SELF)

    # Velocity-u component Vec
    vec_u_comp = da_scalar.createGlobalVec()
    arr_u = da_scalar.getVecArray(vec_u_comp)
    arr_u[:, :] = u_np

    # Velocity-v component Vec
    vec_v_comp = da_scalar.createGlobalVec()
    arr_v = da_scalar.getVecArray(vec_v_comp)
    arr_v[:, :] = v_np

    # Pressure Vec
    vec_pres = da_scalar.createGlobalVec()
    p_arr = solver.pressure_da.getVecArray(p)
    p_np = p_arr[:, :].copy()
    arr_p = da_scalar.getVecArray(vec_pres)
    arr_p[:, :] = p_np

    # DRRN error magnitude Vec
    vec_err = da_scalar.createGlobalVec()
    arr_e = da_scalar.getVecArray(vec_err)
    arr_e[:, :] = err_mag

    write_vtk(
        da_scalar,
        fields={
            "velocity_u": vec_u_comp,
            "velocity_v": vec_v_comp,
            "pressure": vec_pres,
            "drrn_error": vec_err,
        },
        filename="cavity_Re500",
        timestep=step,
    )

    print(f"  step {step+1:3d}  max_err={err_mag.max():.4f}  mean_err={err_mag.mean():.4f}")

print("VTK files written to outputs/vtk/")
