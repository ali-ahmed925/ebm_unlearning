from __future__ import annotations

import torch
from torch import nn


def forget_loss(E: nn.Module, xf: torch.Tensor, yf: torch.Tensor, y_neg: torch.Tensor, m: float) -> torch.Tensor:
    """
    Bounded margin forget (contrastive):
        mean( relu(m - (E(xf,yf) - E(xf,y_neg))) )
      = mean( relu(m + E(xf,y_neg) - E(xf,yf)) )
    """
    e_pos = E(xf, yf)
    e_neg = E(xf, y_neg)
    return torch.relu(float(m) + e_neg - e_pos).mean()


