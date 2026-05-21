"""Physics-informed losses for DRRN supervision using finite-difference divergence penalties."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def divergence_loss(
    u_fine_pred: torch.Tensor,
    dx: float,
    dy: float,
    padding_mode: str = "dirichlet",
) -> torch.Tensor:
    """
    Compute the mean squared divergence of a predicted velocity field.

    `u_fine_pred` must have shape `(B, 2, H, W)` with channel 0 = u and channel 1 = v.
    """

    if u_fine_pred.ndim != 4 or u_fine_pred.shape[1] != 2:
        raise ValueError(f"Expected `(B, 2, H, W)`, received {tuple(u_fine_pred.shape)}.")

    u = u_fine_pred[:, 0, :, :]
    v = u_fine_pred[:, 1, :, :]
    mode = padding_mode.lower()

    if mode == "periodic":
        du_dx = (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dx)
        dv_dy = (torch.roll(v, shifts=-1, dims=-2) - torch.roll(v, shifts=1, dims=-2)) / (2.0 * dy)
        divergence = du_dx + dv_dy
    elif mode == "dirichlet":
        divergence = torch.zeros_like(u)
        divergence[:, 1:-1, 1:-1] = (
            (u[:, 1:-1, 1:-1] - u[:, 1:-1, :-2]) / dx
            + (v[:, 1:-1, 1:-1] - v[:, :-2, 1:-1]) / dy
        )
    else:
        raise ValueError(f"Unsupported padding_mode {padding_mode!r}.")

    return divergence.square().mean()


def physics_informed_loss(
    u_fine_pred: torch.Tensor,
    u_fine_true: torch.Tensor,
    lambda_div: float,
    dx: float,
    dy: float,
    padding_mode: str = "dirichlet",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine reconstruction and divergence losses into the DRRN training objective."""

    reconstruction = F.mse_loss(u_fine_pred, u_fine_true)
    divergence = divergence_loss(u_fine_pred=u_fine_pred, dx=dx, dy=dy, padding_mode=padding_mode)
    total = reconstruction + lambda_div * divergence
    return total, reconstruction, divergence
