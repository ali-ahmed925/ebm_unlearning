from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ebm_unlearning.src.losses.forget import forget_loss
from ebm_unlearning.src.losses.energy_reg import energy_l2
from ebm_unlearning.src.losses.margin import margin_loss
from ebm_unlearning.src.losses.retain import retain_loss


@dataclass(frozen=True)
class LossWeights:
    lambda_f: float = 1.0
    lambda_r: float = 10.0
    lambda_m: float = 1.0
    lambda_e: float = 1.0e-3


def total_unlearning_loss(
    E: nn.Module,
    E0: nn.Module,
    xf: torch.Tensor,
    yf: torch.Tensor,
    y_neg_f: torch.Tensor,
    xr: torch.Tensor,
    yr: torch.Tensor,
    *,
    margin: float,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    lf = forget_loss(E, xf, yf, y_neg_f, float(margin))
    lr = retain_loss(E, E0, xr, yr)
    lm = margin_loss(E, xf, yf, xr, yr, margin)
    le_f = energy_l2(E, xf, yf)
    le_r = energy_l2(E, xr, yr)
    le = 0.5 * (le_f + le_r)

    total = weights.lambda_f * lf + weights.lambda_r * lr + weights.lambda_m * lm + weights.lambda_e * le
    return total, {"forget": lf.detach(), "retain": lr.detach(), "margin": lm.detach(), "energy_reg": le.detach()}


