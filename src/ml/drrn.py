"""EDSR-inspired DRRN for coarse-to-fine velocity field reconstruction under tight VRAM limits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """Two-convolution residual block without batch normalization."""

    def __init__(self, n_feats: int, res_scale: float) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class DRRN(nn.Module):
    """Upsample `(u, v)` velocity fields with a lightweight EDSR-style architecture."""

    def __init__(
        self,
        n_resblocks: int = 8,
        n_feats: int = 32,
        scale: int = 4,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError("DRRN scale must be at least 1.")

        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = scale
        self.res_scale = res_scale

        self.head = nn.Conv2d(2, n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *(ResidualBlock(n_feats=n_feats, res_scale=res_scale) for _ in range(n_resblocks))
        )
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)
        self.tail = nn.Sequential(*self._make_upsampler(scale, n_feats), nn.Conv2d(n_feats, 2, 3, 1, 1))

    def _make_upsampler(self, scale: int, n_feats: int) -> list[nn.Module]:
        layers: list[nn.Module] = []
        if scale == 1:
            return layers
        if scale in {2, 4, 8}:
            for _ in range(int(math.log2(scale))):
                layers.append(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1))
                layers.append(nn.PixelShuffle(2))
            return layers
        if scale == 3:
            return [nn.Conv2d(n_feats, n_feats * 9, kernel_size=3, padding=1), nn.PixelShuffle(3)]
        raise ValueError("DRRN supports scale factors 2, 3, 4, or 8.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        residual = self.body_conv(self.body(features))
        trunk = features + residual
        refined = self.tail(trunk)
        if self.scale == 1:
            skip = x
        else:
            skip = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return skip + refined

    def count_parameters(self) -> tuple[int, int]:
        """Print and return total/trainable parameter counts."""

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        print(f"DRRN parameters: total={total}, trainable={trainable}")
        return total, trainable
