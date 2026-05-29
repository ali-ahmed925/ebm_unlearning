from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ClassificationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    confusion: np.ndarray

    overall_accuracy: float
    per_class_accuracy: Dict[int, float]

    forget_accuracy: Optional[float] = None
    retain_accuracy: Optional[float] = None


@torch.no_grad()
def predict_argmin_energy(
    E: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    y_chunk: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict via:
        y_hat(x) = argmin_y E(x, y)
    without using labels at inference.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if y_chunk < 1:
        raise ValueError("y_chunk must be >= 1")

    E.eval()
    yt: list[np.ndarray] = []
    yp: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).to(torch.long)
        b = int(xb.shape[0])

        # Compute energies for all labels in chunks to control memory.
        all_e: list[torch.Tensor] = []
        for y0 in range(0, int(num_classes), int(y_chunk)):
            ys = torch.arange(y0, min(y0 + int(y_chunk), int(num_classes)), device=device, dtype=torch.long)
            # (B, C, ...)
            x_rep = xb.unsqueeze(1).expand(b, ys.numel(), *xb.shape[1:]).reshape(b * ys.numel(), *xb.shape[1:])
            y_rep = ys.unsqueeze(0).expand(b, ys.numel()).reshape(b * ys.numel())
            e = E(x_rep, y_rep).reshape(b, ys.numel())  # (B, chunk)
            all_e.append(e)

        e_all = torch.cat(all_e, dim=1)  # (B, num_classes)
        y_hat = torch.argmin(e_all, dim=1)  # (B,)

        yt.append(yb.detach().cpu().numpy())
        yp.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(yt, axis=0) if yt else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(yp, axis=0) if yp else np.empty((0,), dtype=np.int64)
    return y_true, y_pred


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in range(int(num_classes)):
        mask = y_true == c
        out[c] = float(np.mean(y_pred[mask] == c)) if np.any(mask) else float("nan")
    return out


def evaluate_classification(
    E: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    forget_label: Optional[int] = None,
    y_chunk: int = 10,
) -> ClassificationResult:
    y_true, y_pred = predict_argmin_energy(E, loader, device=device, num_classes=num_classes, y_chunk=y_chunk)
    conf = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
    overall = float(np.mean(y_true == y_pred)) if y_true.size else float("nan")
    per_cls = _per_class_accuracy(y_true, y_pred, num_classes=num_classes)

    forget_acc = None
    retain_acc = None
    if forget_label is not None and y_true.size:
        f = int(forget_label)
        f_mask = y_true == f
        r_mask = ~f_mask
        forget_acc = float(np.mean(y_pred[f_mask] == f)) if np.any(f_mask) else float("nan")
        retain_acc = float(np.mean(y_true[r_mask] == y_pred[r_mask])) if np.any(r_mask) else float("nan")

    return ClassificationResult(
        y_true=y_true,
        y_pred=y_pred,
        confusion=conf,
        overall_accuracy=overall,
        per_class_accuracy=per_cls,
        forget_accuracy=forget_acc,
        retain_accuracy=retain_acc,
    )


