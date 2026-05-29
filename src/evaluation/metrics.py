from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EnergyStats:
    mean: float
    std: float


@torch.no_grad()
def collect_energies(E: nn.Module, loader: DataLoader, *, device: torch.device) -> np.ndarray:
    E.eval()
    out: list[np.ndarray] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).to(torch.long)
        e = E(xb, yb).detach().float().cpu().numpy()
        out.append(e)
    if not out:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out, axis=0)


def _stats(arr: np.ndarray) -> EnergyStats:
    if arr.size == 0:
        return EnergyStats(mean=float("nan"), std=float("nan"))
    return EnergyStats(mean=float(arr.mean()), std=float(arr.std(ddof=0)))


def energy_gap(forget_e: np.ndarray, retain_e: np.ndarray) -> float:
    if forget_e.size == 0 or retain_e.size == 0:
        return float("nan")
    return float(forget_e.mean() - retain_e.mean())


def forgetting_score(forget_e_after: np.ndarray, forget_e_before: np.ndarray) -> float:
    if forget_e_after.size == 0 or forget_e_before.size == 0:
        return float("nan")
    return float(forget_e_after.mean() - forget_e_before.mean())


def retention_score(retain_e_after: np.ndarray, retain_e_before: np.ndarray) -> float:
    if retain_e_after.size == 0 or retain_e_before.size == 0:
        return float("nan")
    return float(np.mean((retain_e_after - retain_e_before) ** 2))


def membership_inference_proxy(train_e: np.ndarray, holdout_e: np.ndarray) -> float:
    """
    Proxy MIA: how separable train vs holdout energies are (AUC).
    0.5 ~ indistinguishable, 1.0 ~ perfectly separable.
    """
    if train_e.size == 0 or holdout_e.size == 0:
        return float("nan")
    x = np.concatenate([train_e, holdout_e], axis=0)
    y = np.concatenate([np.ones_like(train_e), np.zeros_like(holdout_e)], axis=0)
    return float(roc_auc_score(y, x))


@torch.no_grad()
def evaluate(
    E_after: nn.Module,
    E_before: nn.Module,
    *,
    forget_train_loader: DataLoader,
    forget_holdout_loader: DataLoader,
    retain_train_loader: DataLoader,
    retain_holdout_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    f_tr_after = collect_energies(E_after, forget_train_loader, device=device)
    f_ho_after = collect_energies(E_after, forget_holdout_loader, device=device)
    r_tr_after = collect_energies(E_after, retain_train_loader, device=device)
    r_ho_after = collect_energies(E_after, retain_holdout_loader, device=device)

    f_ho_before = collect_energies(E_before, forget_holdout_loader, device=device)
    r_ho_before = collect_energies(E_before, retain_holdout_loader, device=device)

    return {
        "forget_energy_mean_after": _stats(f_ho_after).mean,
        "forget_energy_mean_before": _stats(f_ho_before).mean,
        "retain_energy_mean_after": _stats(r_ho_after).mean,
        "retain_energy_mean_before": _stats(r_ho_before).mean,
        "energy_gap_after": energy_gap(f_ho_after, r_ho_after),
        "forgetting_score": forgetting_score(f_ho_after, f_ho_before),
        "retention_score": retention_score(r_ho_after, r_ho_before),
        "mia_proxy_forget_auc": membership_inference_proxy(f_tr_after, f_ho_after),
        "mia_proxy_retain_auc": membership_inference_proxy(r_tr_after, r_ho_after),
    }


