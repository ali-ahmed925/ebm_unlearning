from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ebm_unlearning.src.losses.total import LossWeights, total_unlearning_loss
from ebm_unlearning.src.utils.tracking import Tracker


@dataclass(frozen=True)
class UnlearnConfig:
    steps: int = 500
    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_f: float = 1.0
    lambda_r: float = 10.0
    lambda_m: float = 1.0
    lambda_e: float = 1.0e-3
    margin: float = 5.0
    neg_k_forget: int = 0   # 0 = use all C-1 negatives; >0 = sample this many randomly
    log_every: int = 50
    checkpoint_path: str = "outputs/checkpoints/ebm_unlearned.pt"


def _cycle(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _freeze(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def unlearn(
    model: nn.Module,
    pretrained: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    *,
    device: torch.device,
    cfg: UnlearnConfig,
    logger,
    tracker: Tracker | None = None,
    seed: int = 0,
    forget_holdout_loader: DataLoader | None = None,
    retain_holdout_loader: DataLoader | None = None,
) -> nn.Module:
    model = model.to(device)
    model.train()
    pretrained = pretrained.to(device)
    _freeze(pretrained)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    weights = LossWeights(
        lambda_f=float(cfg.lambda_f),
        lambda_r=float(cfg.lambda_r),
        lambda_m=float(cfg.lambda_m),
        lambda_e=float(cfg.lambda_e),
    )

    f_iter = _cycle(forget_loader)
    r_iter = _cycle(retain_loader)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    f_ho_iter = _cycle(forget_holdout_loader) if forget_holdout_loader is not None else None
    r_ho_iter = _cycle(retain_holdout_loader) if retain_holdout_loader is not None else None

    def _mia_auc(train_e: torch.Tensor, holdout_e: torch.Tensor) -> float:
        x = torch.cat([train_e, holdout_e], dim=0).detach().float().cpu().numpy()
        y = torch.cat([torch.ones_like(train_e), torch.zeros_like(holdout_e)], dim=0).detach().cpu().numpy()
        return float(roc_auc_score(y, x))

    for step in range(int(cfg.steps)):
        xf, yf = next(f_iter)
        xr, yr = next(r_iter)
        xf = xf.to(device)
        yf = yf.to(device).to(torch.long)
        xr = xr.to(device)
        yr = yr.to(device).to(torch.long)

        if not hasattr(model, "label_emb"):
            raise ValueError("Expected a label-conditioned EnergyModel with `label_emb`.")
        num_classes = int(model.label_emb.num_embeddings)
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        # build negatives for forget loss
        k = int(cfg.neg_k_forget) if int(cfg.neg_k_forget) > 0 else (num_classes - 1)
        if k >= num_classes - 1:
            # all C-1 negatives: (B, C-1)
            all_labels = torch.arange(num_classes, device=device).unsqueeze(0).expand(yf.shape[0], -1)
            mask = all_labels != yf.unsqueeze(1)
            y_neg_f = all_labels[mask].view(yf.shape[0], num_classes - 1)
        else:
            # sample k random negatives: (B, k)
            rows = []
            for _ in range(k):
                r = torch.randint(0, num_classes - 1, size=yf.shape, generator=g, device="cpu", dtype=torch.long).to(device)
                rows.append(r + (r >= yf).to(torch.long))
            y_neg_f = torch.stack(rows, dim=1)   # (B, k)

        total, parts = total_unlearning_loss(
            model,
            pretrained,
            xf,
            yf,
            y_neg_f,
            xr,
            yr,
            margin=float(cfg.margin),
            weights=weights,
        )

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if tracker is not None:
            tracker.log_scalar("unlearn/total", float(total.item()), step)
            tracker.log_scalars(
                "unlearn/loss",
                {k: float(v.item()) for k, v in parts.items()},
                step,
            )
            with torch.no_grad():
                ef = model(xf, yf)
                er = model(xr, yr)
                ef0 = pretrained(xf, yf)
                er0 = pretrained(xr, yr)
                y_neg_log = torch.randint(0, num_classes - 1, size=yf.shape, generator=g, device="cpu", dtype=torch.long).to(device)
                y_neg_log = y_neg_log + (y_neg_log >= yf).to(torch.long)
                gap_fw = (ef - model(xf, y_neg_log)).mean()
                tracker.log_scalar("unlearn/energy_forget_mean", float(ef.mean().item()), step)
                tracker.log_scalar("unlearn/energy_retain_mean", float(er.mean().item()), step)
                tracker.log_scalar("unlearn/energy_gap_forget_minus_retain", float((ef.mean() - er.mean()).item()), step)
                tracker.log_scalar("unlearn/forgetting_score_batch", float((ef.mean() - ef0.mean()).item()), step)
                tracker.log_scalar("unlearn/retention_score_batch", float(((er - er0) ** 2).mean().item()), step)
                tracker.log_scalar("unlearn/forget_gap_correct_minus_wrong", float(gap_fw.item()), step)
                tracker.log_scalar("unlearn/margin_effective", float(cfg.margin), step)

                if f_ho_iter is not None:
                    xf_ho, yf_ho = next(f_ho_iter)
                    xf_ho = xf_ho.to(device)
                    yf_ho = yf_ho.to(device).to(torch.long)
                    e_f_tr = model(xf, yf).detach()
                    e_f_ho = model(xf_ho, yf_ho).detach()
                    tracker.log_scalar("unlearn/mia_proxy_forget_auc_batch", _mia_auc(e_f_tr, e_f_ho), step)

                if r_ho_iter is not None:
                    xr_ho, yr_ho = next(r_ho_iter)
                    xr_ho = xr_ho.to(device)
                    yr_ho = yr_ho.to(device).to(torch.long)
                    e_r_tr = model(xr, yr).detach()
                    e_r_ho = model(xr_ho, yr_ho).detach()
                    tracker.log_scalar("unlearn/mia_proxy_retain_auc_batch", _mia_auc(e_r_tr, e_r_ho), step)

        if step % int(cfg.log_every) == 0:
            with torch.no_grad():
                y_neg_log = torch.randint(0, num_classes - 1, size=yf.shape, generator=g, device="cpu", dtype=torch.long).to(device)
                y_neg_log = y_neg_log + (y_neg_log >= yf).to(torch.long)
                gap_fw = (model(xf, yf) - model(xf, y_neg_log)).mean()
            logger.info(
                "[unlearn] step=%d margin=%.4f gap_fw=%.4f total=%.6f forget=%.6f retain=%.6f margin_loss=%.6f energy_reg=%.6f",
                step,
                float(cfg.margin),
                float(gap_fw.item()),
                float(total.item()),
                float(parts["forget"].item()),
                float(parts["retain"].item()),
                float(parts["margin"].item()),
                float(parts["energy_reg"].item()),
            )

    if cfg.checkpoint_path:
        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
        torch.save({"model": model.state_dict()}, cfg.checkpoint_path)
        logger.info(f"[unlearn] saved checkpoint to {cfg.checkpoint_path}")

    return model


