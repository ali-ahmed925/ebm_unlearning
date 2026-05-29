from __future__ import annotations

import torch
from torch import nn


def energy_l2(E: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy magnitude regularization: mean(E(x,y)^2)."""
    e = E(x, y)
    return (e**2).mean()


