"""Bridge PETSc DMDA velocity vectors into PyTorch tensors for DRRN inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

try:
    from petsc4py import PETSc
except ImportError:  # PETSC_STUB
    PETSc = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from petsc4py import PETSc as PETScTyping


def _require_petsc() -> None:
    if PETSc is None:  # pragma: no cover - covered by PETSc-backed tests
        raise RuntimeError("petsc4py is required for the PETSc/Torch bridge.")


def _grid_shape_from_da(da: "PETScTyping.DMDA") -> tuple[int, int]:
    sizes = tuple(int(value) for value in da.getSizes())
    if len(sizes) < 2:
        raise ValueError(f"Expected a 2D DMDA, received sizes={sizes!r}.")
    nx, ny = sizes[:2]
    return ny, nx


def _unpack_corners(corners: Sequence[object]) -> tuple[int, int, int, int]:
    if len(corners) == 2:
        starts = tuple(int(value) for value in corners[0])  # type: ignore[arg-type]
        widths = tuple(int(value) for value in corners[1])  # type: ignore[arg-type]
        if len(starts) < 2 or len(widths) < 2:
            raise ValueError(f"Unsupported DMDA corner format: {corners!r}")
        return starts[0], starts[1], widths[0], widths[1]
    if len(corners) >= 4:
        xs, ys, xm, ym = (int(corners[index]) for index in range(4))
        return xs, ys, xm, ym
    raise ValueError(f"Unsupported DMDA corner format: {corners!r}")


def _require_velocity_dof(da: "PETScTyping.DMDA") -> int:
    dof = int(da.getDof())
    if dof != 2:
        raise ValueError(f"Expected a velocity DMDA with 2 dof, received dof={dof}.")
    return dof


def vec_to_tensor(vec: "PETScTyping.Vec", device: str) -> torch.Tensor:
    """Return a `(1, 2, H, W)` tensor view/copy of a PETSc DMDA velocity vector."""

    _require_petsc()
    da = vec.getDM()
    if da is None:
        raise ValueError("PETSc Vec is not attached to a DMDA; cannot infer H/W.")

    height, width = _grid_shape_from_da(da)
    dof = _require_velocity_dof(da)
    array_view = vec.getArray(readonly=True)
    tensor = (
        torch.as_tensor(array_view)
        .reshape(height, width, dof)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    if device.startswith("cuda"):
        cpu_tensor = tensor
        if not cpu_tensor.is_pinned():
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(device, non_blocking=True)
    return tensor


def tensor_to_vec(tensor: torch.Tensor, vec: "PETScTyping.Vec") -> None:
    """Write a `(1, 2, H, W)` tensor back into a PETSc DMDA velocity vector."""

    _require_petsc()
    da = vec.getDM()
    if da is None:
        raise ValueError("PETSc Vec is not attached to a DMDA; cannot validate shape.")

    height, width = _grid_shape_from_da(da)
    dof = _require_velocity_dof(da)
    expected_shape = (1, dof, height, width)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected tensor shape {expected_shape}, received {tuple(tensor.shape)}.")

    if tensor.is_cuda:
        torch.cuda.synchronize(device=tensor.device)
    host_tensor = tensor.detach().to("cpu").contiguous()
    array_view = host_tensor.squeeze(0).permute(1, 2, 0).numpy().reshape(height * width * dof)
    vec.setArray(array_view)


def enforce_bcs(vec: "PETScTyping.Vec", da: "PETScTyping.DMDA", bc_type: str) -> None:
    """Apply physical boundary conditions directly on the PETSc vector storage."""

    _require_petsc()
    bc_type_normalized = bc_type.lower()
    if bc_type_normalized not in {"dirichlet", "neumann", "periodic", "lid_driven_cavity"}:
        raise ValueError(f"Unsupported boundary condition {bc_type!r}.")

    height, width = _grid_shape_from_da(da)
    _require_velocity_dof(da)
    xs, ys, xm, ym = _unpack_corners(da.getCorners())
    array_view = da.getVecArray(vec)

    touches_left = xs == 0
    touches_right = xs + xm == width
    touches_bottom = ys == 0
    touches_top = ys + ym == height

    if bc_type_normalized == "dirichlet":
        if touches_bottom:
            array_view[0, :, :] = 0.0
        if touches_top:
            array_view[-1, :, :] = 0.0
        if touches_left:
            array_view[:, 0, :] = 0.0
        if touches_right:
            array_view[:, -1, :] = 0.0
        return

    if bc_type_normalized == "lid_driven_cavity":
        if touches_bottom:
            array_view[0, :, :] = 0.0
        if touches_left:
            array_view[:, 0, :] = 0.0
        if touches_right:
            array_view[:, -1, :] = 0.0
        if touches_top:
            array_view[-1, :, 0] = 1.0
            array_view[-1, :, 1] = 0.0
        return

    if bc_type_normalized == "neumann":
        if touches_bottom and ym > 1:
            array_view[0, :, :] = array_view[1, :, :]
        if touches_top and ym > 1:
            array_view[-1, :, :] = array_view[-2, :, :]
        if touches_left and xm > 1:
            array_view[:, 0, :] = array_view[:, 1, :]
        if touches_right and xm > 1:
            array_view[:, -1, :] = array_view[:, -2, :]
        return

    # For periodic DMDAs in serial, mirror the opposing physical edge values.
    if touches_bottom and touches_top and ym > 1:
        array_view[0, :, :] = array_view[-2, :, :]
        array_view[-1, :, :] = array_view[1, :, :]
    if touches_left and touches_right and xm > 1:
        array_view[:, 0, :] = array_view[:, -2, :]
        array_view[:, -1, :] = array_view[:, 1, :]
