from __future__ import annotations

import torch
from torch import nn


def margin_loss(
    E: nn.Module, xf: torch.Tensor, yf: torch.Tensor, xr: torch.Tensor, yr: torch.Tensor, m: float
) -> torch.Tensor:
    """
    Soft margin (logistic), never fully goes to zero:
        mean( softplus(m + E(xr,yr) - E(xf,yf)) )
    """
    ef = E(xf, yf)
    er = E(xr, yr)
    return torch.nn.functional.softplus(float(m) + er - ef).mean()


