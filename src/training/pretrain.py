from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ebm_unlearning.src.losses.pretrain import supervised_energy_contrast_loss
from ebm_unlearning.src.evaluation.classification import evaluate_classification
from ebm_unlearning.src.utils.tracking import Tracker


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    margin: float = 1.0
    k_neg: int = 5
    neg_chunk: int = 10
    log_every: int = 100
    checkpoint_path: str = "outputs/checkpoints/ebm_pretrained.pt"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 1.0e-3
    mode: str = "max"  # max|min


def _sample_negative_labels(y: torch.Tensor, *, num_classes: int, generator: torch.Generator) -> torch.Tensor:
    """
    Sample y' != y uniformly by drawing from [0, num_classes-2] and shifting past y.
    """
    if y.dtype != torch.long:
        y = y.to(torch.long)
    k = int(num_classes)
    if k < 2:
        raise ValueError("num_classes must be >= 2")
    # Sample on CPU to match the (CPU) generator, then move to y.device.
    y_cpu = y.detach().to("cpu")
    r = torch.randint(low=0, high=k - 1, size=y_cpu.shape, generator=generator, device="cpu", dtype=torch.long)
    y_neg = r + (r >= y_cpu).to(torch.long)
    return y_neg.to(device=y.device)


def _sample_negative_labels_k(
    y: torch.Tensor, *, num_classes: int, k_neg: int, generator: torch.Generator
) -> torch.Tensor:
    if k_neg < 1:
        raise ValueError("k_neg must be >= 1")
    if y.dtype != torch.long:
        y = y.to(torch.long)
    k = int(num_classes)
    if k < 2:
        raise ValueError("num_classes must be >= 2")
    y_cpu = y.detach().to("cpu").view(-1, 1)
    r = torch.randint(
        low=0,
        high=k - 1,
        size=(int(y_cpu.shape[0]), int(k_neg)),
        generator=generator,
        device="cpu",
        dtype=torch.long,
    )
    y_neg = r + (r >= y_cpu).to(torch.long)
    return y_neg.to(device=y.device)


def pretrain_ebm(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    cfg: PretrainConfig,
    logger,
    seed: int,
    tracker: Tracker | None = None,
    val_loader: DataLoader | None = None,
    early_stopping: EarlyStoppingConfig | None = None,
) -> nn.Module:
    model = model.to(device)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    step = 0
    es = early_stopping or EarlyStoppingConfig(enabled=False)
    best = -float("inf") if es.mode == "max" else float("inf")
    bad_epochs = 0
    for epoch in range(int(cfg.epochs)):
        pbar = tqdm(loader, desc=f"pretrain epoch {epoch+1}/{cfg.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device).to(torch.long)

            if not hasattr(model, "label_emb"):
                raise ValueError("Expected a label-conditioned EnergyModel with `label_emb`.")
            num_classes = int(model.label_emb.num_embeddings)

            y_neg = _sample_negative_labels_k(yb, num_classes=num_classes, k_neg=int(cfg.k_neg), generator=g)
            loss = supervised_energy_contrast_loss(
                model,
                xb,
                yb,
                y_neg,
                float(cfg.margin),
                neg_chunk=int(cfg.neg_chunk),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if tracker is not None:
                tracker.log_scalar("pretrain/loss", float(loss.item()), step)
                with torch.no_grad():
                    tracker.log_scalar("pretrain/energy_pos_mean", float(model(xb, yb).mean().item()), step)
                    tracker.log_scalar("pretrain/energy_neg_mean", float(model(xb, y_neg[:, 0]).mean().item()), step)

            if step % int(cfg.log_every) == 0:
                logger.info(f"[pretrain] step={step} loss={loss.item():.6f}")
            step += 1

        if val_loader is not None:
            res = evaluate_classification(
                model,
                val_loader,
                device=device,
                num_classes=int(model.label_emb.num_embeddings),
                forget_label=None,
                y_chunk=5,
            )
            val_acc = float(res.overall_accuracy)
            if tracker is not None:
                tracker.log_scalar("pretrain/val_overall_acc", val_acc, step)
            logger.info(f"[pretrain] epoch_end={epoch+1} val_overall_acc={val_acc:.6f}")

            if es.enabled:
                improved = (val_acc > best + float(es.min_delta)) if es.mode == "max" else (val_acc < best - float(es.min_delta))
                if improved:
                    best = val_acc
                    bad_epochs = 0
                    if cfg.checkpoint_path:
                        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
                        torch.save({"model": model.state_dict()}, cfg.checkpoint_path)
                        logger.info(f"[pretrain] saved best checkpoint to {cfg.checkpoint_path}")
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(es.patience):
                        logger.info(f"[pretrain] early stopping at epoch {epoch+1} (best={best:.6f})")
                        return model

    if cfg.checkpoint_path and (val_loader is None or not es.enabled):
        os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
        torch.save({"model": model.state_dict()}, cfg.checkpoint_path)
        logger.info(f"[pretrain] saved checkpoint to {cfg.checkpoint_path}")

    return model


def load_pretrained(model: nn.Module, checkpoint_path: str, *, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model.to(device)


