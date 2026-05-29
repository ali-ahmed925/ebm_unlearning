from __future__ import annotations

import torch
from torch import nn


def retain_loss(E: nn.Module, E0: nn.Module, xr: torch.Tensor, yr: torch.Tensor) -> torch.Tensor:
    """Energy preservation: match current energies to pretrained energies on retain data."""
    er = E(xr, yr)
    with torch.no_grad():
        er0 = E0(xr, yr)
        scale = er0.abs().mean().clamp_min(1.0e-6)
    return torch.mean((er - er0) ** 2) / scale


