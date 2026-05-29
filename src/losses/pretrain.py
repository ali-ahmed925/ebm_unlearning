from __future__ import annotations

import torch
from torch import nn


def supervised_energy_contrast_loss(
    E: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    y_neg: torch.Tensor,
    m: float,
    *,
    neg_chunk: int = 10,
) -> torch.Tensor:
    """
    Supervised energy contrast:
      If y_neg is shape (B,): mean( relu(m + E(x,y) - E(x,y_neg)) )
      If y_neg is shape (B,K): mean( max_k relu(m + E(x,y) - E(x,y_neg_k)) )
    """
    if neg_chunk < 1:
        raise ValueError("neg_chunk must be >= 1")

    e_pos = E(x, y)  # (B,)
    if y_neg.ndim == 1:
        e_neg = E(x, y_neg)  # (B,)
        return torch.relu(float(m) + e_pos - e_neg).mean()

    if y_neg.ndim != 2:
        raise ValueError("y_neg must have shape (B,) or (B,K)")

    b, k = int(y_neg.shape[0]), int(y_neg.shape[1])
    losses: list[torch.Tensor] = []
    for j in range(0, k, int(neg_chunk)):
        y_chunk = y_neg[:, j : j + int(neg_chunk)]
        kc = int(y_chunk.shape[1])
        x_rep = x.unsqueeze(1).expand(b, kc, *x.shape[1:]).reshape(b * kc, *x.shape[1:])
        y_flat = y_chunk.reshape(b * kc)
        e_neg = E(x_rep, y_flat).reshape(b, kc)  # (B,kc)
        l = torch.relu(float(m) + e_pos.unsqueeze(1) - e_neg)  # (B,kc)
        losses.append(l)

    l_all = torch.cat(losses, dim=1)  # (B,K)
    return l_all.max(dim=1).values.mean()


