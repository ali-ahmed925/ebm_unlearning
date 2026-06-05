from __future__ import annotations

import torch
from torch import nn


def forget_loss(E: nn.Module, xf: torch.Tensor, yf: torch.Tensor, y_neg: torch.Tensor, m: float) -> torch.Tensor:
    """
    Bounded margin forget (contrastive):
        y_neg (B,):   mean( softplus(m + E(xf,y_neg) - E(xf,yf)) )
        y_neg (B,K):  mean( max_k softplus(m + E(xf,y_neg_k) - E(xf,yf)) )

    softplus: never exactly zero, maintains gradient even for satisfied constraints.
    With y_neg (B,K), saturation implies E(xf,yf) > E(xf,y') + m for all provided y'.
    """
    e_pos = E(xf, yf)
    if y_neg.dim() == 1:
        e_neg = E(xf, y_neg)
        return torch.nn.functional.softplus(float(m) + e_neg - e_pos).mean()
    B, K = int(y_neg.shape[0]), int(y_neg.shape[1])
    xf_rep = xf.unsqueeze(1).expand(-1, K, *xf.shape[1:]).reshape(B * K, *xf.shape[1:])
    e_neg = E(xf_rep, y_neg.reshape(B * K)).view(B, K)
    return torch.nn.functional.softplus(float(m) + e_neg - e_pos.unsqueeze(1)).max(dim=1).values.mean()


