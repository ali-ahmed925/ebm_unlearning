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
    lambda_clip: float = 0.0      # CLIP subspace generalization loss weight
    n_pca_components: int = 10    # number of PCA directions for forget subspace
    margin: float = 5.0
    log_every: int = 50
    checkpoint_path: str = "outputs/checkpoints/ebm_unlearned.pt"


def _cycle(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def _freeze(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def _sample_neg(y: torch.Tensor, num_classes: int, g: torch.Generator, device: torch.device) -> torch.Tensor:
    r = torch.randint(0, num_classes - 1, size=y.shape, generator=g, device="cpu", dtype=torch.long).to(device)
    return r + (r >= y).to(torch.long)


def _all_negatives(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    """Return all C-1 negative labels per sample: shape (B, C-1)."""
    all_labels = torch.arange(num_classes, device=device).unsqueeze(0).expand(y.shape[0], -1)  # (B, C)
    mask = all_labels != y.unsqueeze(1)                                                          # (B, C)
    return all_labels[mask].view(y.shape[0], num_classes - 1)                                   # (B, C-1)


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
    import copy
    model = model.to(device)
    model.train()
    pretrained = copy.deepcopy(pretrained).to(device)
    _freeze(pretrained)

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    weights = LossWeights(
        lambda_f=float(cfg.lambda_f),
        lambda_r=float(cfg.lambda_r),
        lambda_m=float(cfg.lambda_m),
        lambda_e=float(cfg.lambda_e),
    )

    use_clip = float(cfg.lambda_clip) > 0
    f_iter   = _cycle(forget_loader)
    r_iter   = _cycle(retain_loader)
    g        = torch.Generator(device="cpu").manual_seed(int(seed))
    f_ho_iter = _cycle(forget_holdout_loader) if forget_holdout_loader is not None else None
    r_ho_iter = _cycle(retain_holdout_loader) if retain_holdout_loader is not None else None

    def _mia_auc(train_e: torch.Tensor, holdout_e: torch.Tensor) -> float:
        x = torch.cat([train_e, holdout_e], dim=0).detach().float().cpu().numpy()
        y = torch.cat([torch.ones_like(train_e), torch.zeros_like(holdout_e)], dim=0).detach().cpu().numpy()
        return float(roc_auc_score(y, x))

    for step in range(int(cfg.steps)):
        # ── Fetch batches ────────────────────────────────────────────────────
        xf, yf = next(f_iter)
        retain_batch = next(r_iter)

        # Retain loader returns (x, y) normally or (x, y, w) when weighted
        if len(retain_batch) == 3:
            xr, yr, wr = retain_batch
            wr = wr.float().to(device)   # (B,) CLIP subspace weights
        else:
            xr, yr = retain_batch
            wr = None

        xf = xf.to(device);  yf = yf.to(device).to(torch.long)
        xr = xr.to(device);  yr = yr.to(device).to(torch.long)

        if not hasattr(model, "label_emb"):
            raise ValueError("Expected a label-conditioned EnergyModel with `label_emb`.")
        num_classes = int(model.label_emb.num_embeddings)

        y_neg_f = _all_negatives(yf, num_classes, device)

        # ── Standard unlearning losses ───────────────────────────────────────
        total, parts = total_unlearning_loss(
            model, pretrained, xf, yf, y_neg_f, xr, yr,
            margin=float(cfg.margin), weights=weights,
        )

        # ── CLIP subspace generalization loss ────────────────────────────────
        # For retain samples with high projection onto the forget subspace,
        # apply a weighted forget signal. This generalizes forgetting to
        # semantically similar classes without explicit supervision.
        if use_clip and wr is not None and wr.sum() > 0:
            y_neg_r = _sample_neg(yr, num_classes, g, device)
            e_pos = model(xr, yr)
            e_neg = model(xr, y_neg_r)
            per_sample = torch.relu(float(cfg.margin) + e_neg - e_pos)  # (B,)
            l_clip = (wr * per_sample).mean()
            total  = total + float(cfg.lambda_clip) * l_clip
            parts["clip"] = l_clip.detach()

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        # ── Logging / tracking ───────────────────────────────────────────────
        if tracker is not None:
            tracker.log_scalar("unlearn/total", float(total.item()), step)
            tracker.log_scalars("unlearn/loss", {k: float(v.item()) for k, v in parts.items()}, step)
            with torch.no_grad():
                ef   = model(xf, yf);     er   = model(xr, yr)
                ef0  = pretrained(xf, yf); er0  = pretrained(xr, yr)
                y_neg_log = _sample_neg(yf, num_classes, g, device)
                gap_fw = (ef - model(xf, y_neg_log)).mean()
                tracker.log_scalar("unlearn/energy_forget_mean",            float(ef.mean()), step)
                tracker.log_scalar("unlearn/energy_retain_mean",            float(er.mean()), step)
                tracker.log_scalar("unlearn/energy_gap_forget_minus_retain", float((ef.mean() - er.mean())), step)
                tracker.log_scalar("unlearn/forgetting_score_batch",        float((ef.mean() - ef0.mean())), step)
                tracker.log_scalar("unlearn/retention_score_batch",         float(((er - er0) ** 2).mean()), step)
                tracker.log_scalar("unlearn/forget_gap_correct_minus_wrong", float(gap_fw), step)
                if wr is not None:
                    tracker.log_scalar("unlearn/clip_weight_mean_batch", float(wr.mean()), step)

                if f_ho_iter is not None:
                    xf_ho, yf_ho = next(f_ho_iter)
                    xf_ho = xf_ho.to(device); yf_ho = yf_ho.to(device).to(torch.long)
                    tracker.log_scalar("unlearn/mia_proxy_forget_auc_batch",
                                       _mia_auc(model(xf, yf).detach(), model(xf_ho, yf_ho).detach()), step)
                if r_ho_iter is not None:
                    xr_ho, yr_ho = next(r_ho_iter)
                    xr_ho = xr_ho.to(device); yr_ho = yr_ho.to(device).to(torch.long)
                    tracker.log_scalar("unlearn/mia_proxy_retain_auc_batch",
                                       _mia_auc(model(xr, yr).detach(), model(xr_ho, yr_ho).detach()), step)

        if step % int(cfg.log_every) == 0:
            with torch.no_grad():
                y_neg_log = _sample_neg(yf, num_classes, g, device)
                gap_fw = (model(xf, yf) - model(xf, y_neg_log)).mean()
            clip_str = f" clip={parts['clip'].item():.6f}" if "clip" in parts else ""
            logger.info(
                "[unlearn] step=%d gap_fw=%.4f total=%.6f forget=%.6f retain=%.6f margin=%.6f energy_reg=%.6f%s",
                step, float(gap_fw), float(total),
                float(parts["forget"]), float(parts["retain"]),
                float(parts["margin"]), float(parts["energy_reg"]),
                clip_str,
            )

    if cfg.checkpoint_path:
        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
        torch.save({"model": model.state_dict()}, cfg.checkpoint_path)
        logger.info(f"[unlearn] saved checkpoint to {cfg.checkpoint_path}")

    return model
