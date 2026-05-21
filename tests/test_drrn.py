"""Unit tests for the DRRN architecture and divergence-aware losses."""

from __future__ import annotations

import math

import pytest
import torch

from src.ml.drrn import DRRN
from src.ml.physics_loss import divergence_loss


def test_forward_pass_shape() -> None:
    model = DRRN(scale=4)
    inputs = torch.randn(2, 2, 5, 5, dtype=torch.float32)
    outputs = model(inputs)
    assert outputs.shape == (2, 2, 20, 20)


def test_divergence_loss_on_divergence_free_field() -> None:
    size = 64
    coords = torch.arange(size, dtype=torch.float32) * (2.0 * math.pi / size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    u = torch.sin(xx) * torch.cos(yy)
    v = -torch.cos(xx) * torch.sin(yy)
    field = torch.stack((u, v), dim=0).unsqueeze(0)
    dx = dy = float((2.0 * math.pi) / size)
    loss = divergence_loss(field, dx=dx, dy=dy, padding_mode="periodic")
    assert loss.item() < 1e-4


def test_divergence_loss_on_diverging_field() -> None:
    size = 32
    coords = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    field = torch.stack((xx, yy), dim=0).unsqueeze(0)
    dx = dy = float(2.0 / max(size - 1, 1))
    loss = divergence_loss(field, dx=dx, dy=dy, padding_mode="periodic")
    assert loss.item() > 0.1
