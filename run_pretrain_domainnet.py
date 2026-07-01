"""
Self-contained DomainNet pretraining runner (mirrors notebook 06 cells 1-3).

Lets you pretrain the EBM headless (no Jupyter) on any machine, driven by a
single --config. Reproduces the notebook logic exactly: same train/val split,
same PretrainConfig, same early stopping, same pretrain_ebm() call.

Usage (RTX 4090 / high-VRAM, 26-class subset at bs=32 from the config):
    conda run -n myn_again python run_pretrain_domainnet.py \
        --config configs/config_domainnet_subset_hivram.yaml

Writes the pretrained model to pretrain.checkpoint_path from the config
(ebm_pretrained_domainnet_subset.pt for the subset/hivram configs).

Requires the DomainNet dataset present at data.data_dir (not in the repo).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import EarlyStoppingConfig, PretrainConfig, pretrain_ebm
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a DomainNet config yaml (relative to project root or absolute).")
    p.add_argument("--tracker", choices=["null", "tensorboard"], default="null",
                   help="null (default, portable) or tensorboard.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg["seed"]))
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    logger = setup_logger("pretrain", log_file=str(ROOT / "outputs" / "logs" / "pretrain_domainnet.log"))
    if args.tracker == "tensorboard":
        from ebm_unlearning.src.utils.tracking import make_tracker
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        tracker = make_tracker("tensorboard",
                               log_dir=str(ROOT / "outputs" / "tensorboard" / "domainnet" / "pretrain" / run_id))
    else:
        tracker = None

    # ── data + train/val split (mirror notebook cell 1) ──────────────────────
    dset = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
    )

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    val_fraction = float(cfg.get("pretrain", {}).get("val_fraction", 0.1))
    val_n = int(round(len(dset) * val_fraction))
    train_n = len(dset) - val_n
    train_dset, val_dset = torch.utils.data.random_split(
        dset, [train_n, val_n],
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

    loader = DataLoader(
        train_dset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dset, batch_size=16, shuffle=False,
        num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    print(f"train/val sizes: {len(train_dset)} / {len(val_dset)}")
    print(f"Classes ({len(dset.classes)}): {dset.classes}")

    # ── model + configs (mirror notebook cell 2) ─────────────────────────────
    model = EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=int(cfg["model"].get("num_classes", 10)),
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "resnet18")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 2)),
        imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
    )

    pre_cfg = PretrainConfig(
        epochs=int(cfg["pretrain"]["epochs"]),
        lr=float(cfg["pretrain"]["lr"]),
        weight_decay=float(cfg["pretrain"]["weight_decay"]),
        margin=float(cfg["pretrain"].get("margin", 1.0)),
        k_neg=int(cfg["pretrain"].get("k_neg", 10)),
        neg_chunk=int(cfg["pretrain"].get("neg_chunk", 5)),
        log_every=int(cfg["pretrain"]["log_every"]),
        checkpoint_path=str(ROOT / cfg["pretrain"]["checkpoint_path"]),
    )

    es = cfg.get("pretrain", {}).get("early_stopping", {})
    es_cfg = EarlyStoppingConfig(
        enabled=bool(es.get("enabled", True)),
        patience=int(es.get("patience", 8)),
        min_delta=float(es.get("min_delta", 1e-3)),
        mode=str(es.get("mode", "max")),
    )

    print(f"Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
    print(f"batch_size={batch_size}  epochs={pre_cfg.epochs}  num_classes={cfg['model'].get('num_classes')}")
    print(f"Checkpoint -> {pre_cfg.checkpoint_path}")

    # ── train (mirror notebook cell 3) ────────────────────────────────────────
    pretrain_ebm(
        model, loader,
        device=device, cfg=pre_cfg, logger=logger, seed=int(cfg["seed"]),
        tracker=tracker, val_loader=val_loader, early_stopping=es_cfg,
    )
    if tracker is not None:
        tracker.close()
    print(f"\nDone. Pretrained checkpoint saved to {pre_cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
