"""Configuration defaults for DRRN training, continuation, and physics penalties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DRRNConfig:
    """Container for DRRN hyperparameters and continuation schedule settings."""

    n_resblocks: int = 8
    n_feats: int = 32
    scale: int = 4
    res_scale: float = 0.1
    lambda_div: float = 1e-3
    lr: float = 1e-4
    batch_size: int = 4
    warmup_epochs: int = 1000
    finetune_epochs: int = 200
    checkpoint_every: int = 50
    padding_mode: str = "dirichlet"
    coarse_res: int = 32
    fine_res: int = 128
    fine_grid_size: int = 128
    steady_tol: float = 1e-6
    max_steps: int = 6000
    dt: float = 0.005
    Re_start: float = 100.0
    Re_step: float = 100.0
    Re_max: float = 1000.0
