"""Unit tests for the PETSc/Torch tensor bridge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

PETSc = pytest.importorskip("petsc4py.PETSc")

from src.bridge.petsc_torch_bridge import enforce_bcs, tensor_to_vec, vec_to_tensor


def _build_dmda(nx: int = 10, ny: int = 10) -> "PETSc.DMDA":
    return PETSc.DMDA().create([nx, ny], dof=2, stencil_width=1, comm=PETSc.COMM_SELF)


def _fill_velocity_field(da: "PETSc.DMDA", vec: "PETSc.Vec") -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, da.getSizes()[0], endpoint=False)
    y = np.linspace(0.0, 2.0 * np.pi, da.getSizes()[1], endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u = np.sin(xx) * np.cos(yy)
    v = np.cos(xx) * np.sin(yy)
    array_view = da.getVecArray(vec)
    array_view[:, :, 0] = u
    array_view[:, :, 1] = v
    return u, v


def test_vec_tensor_round_trip_with_dirichlet_bcs() -> None:
    da = _build_dmda()
    vec = da.createGlobalVec()
    original_u, original_v = _fill_velocity_field(da, vec)

    tensor = vec_to_tensor(vec, device="cpu")
    assert tensor.shape == (1, 2, 10, 10)

    perturbed = tensor + 1.0
    tensor_to_vec(perturbed, vec)
    enforce_bcs(vec, da, "dirichlet")

    array_view = da.getVecArray(vec)
    expected_u = original_u + 1.0
    expected_v = original_v + 1.0

    np.testing.assert_allclose(array_view[0, :, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(array_view[-1, :, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(array_view[:, 0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(array_view[:, -1, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(array_view[1:-1, 1:-1, 0], expected_u[1:-1, 1:-1], atol=1e-12)
    np.testing.assert_allclose(array_view[1:-1, 1:-1, 1], expected_v[1:-1, 1:-1], atol=1e-12)
