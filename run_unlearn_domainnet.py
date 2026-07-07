"""
Self-contained DomainNet unlearning runner (mirrors notebook 07 cells 1-5).

Why this exists: notebook 07 loads its config in three separate cells, which is
error-prone (the earlier airplane/label mix-ups came from inconsistent cell
edits). This script does the identical pipeline in one shot, driven by a single
--config, so running on another machine can't desync.

It reproduces EXACTLY the notebook logic — same split, same DINO subspace
weights, same masking flag, same UnlearnConfig, same unlearn() call. No method
changes. It only adds a CLI + a --mask-mode switch for the two modes you already
use (the commented/uncommented masking line in the notebook).

Usage (high-VRAM machine, propagation/cross-class run):
    conda run -n myn_again python run_unlearn_domainnet.py \
        --config configs/config_domainnet_subset_hivram.yaml \
        --mask-mode cross-class

Isolated (cross-domain-only, for the Table-1 selective-unlearning result):
    conda run -n myn_again python run_unlearn_domainnet.py \
        --config configs/config_domainnet_subset_hivram.yaml \
        --mask-mode cross-domain

The unlearned checkpoint is written to unlearning.checkpoint_path from the config.
Existing checkpoints are never touched unless the config points at them.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split
from ebm_unlearning.src.losses.clip_subspace import (
    compute_domainnet_subspace_weights,
    load_dino_encoder,
    WeightedSubset,
)
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.training.unlearn import UnlearnConfig, unlearn
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a DomainNet config yaml (relative to project root or absolute).")
    p.add_argument("--mask-mode", choices=["cross-class", "cross-domain"], default="cross-class",
                   help="cross-class: all retain samples get the weighted forget signal (propagation, notebook masking OFF). "
                        "cross-domain: zero out non-forget-class retain samples (isolated, notebook masking ON).")
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

    logger = setup_logger("unlearn", log_file=str(ROOT / "outputs" / "logs" / "unlearn_domainnet.log"))
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.tracker == "tensorboard":
        from ebm_unlearning.src.utils.tracking import make_tracker
        tracker = make_tracker("tensorboard",
                               log_dir=str(ROOT / "outputs" / "tensorboard" / "domainnet" / "unlearn" / run_id))
    else:
        tracker = None

    # ── CELL 1: data + forget/retain split ───────────────────────────────────
    dset = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
    )
    forget_spec = ForgetSpec(
        mode="class_domain",
        class_label=int(cfg["data"]["forget"]["class_label"]),
        domain=str(cfg["data"]["forget"]["domain"]),
    )
    retain_spec = RetainSpec()
    forget_all, retain_all = split_forget_retain(dset, forget_spec, retain_spec)

    holdout_fraction = float(cfg["evaluation"]["holdout_fraction"])
    forget_train, _ = train_holdout_split(forget_all, holdout_fraction, seed=int(cfg["seed"]))
    retain_train, _ = train_holdout_split(retain_all, holdout_fraction, seed=int(cfg["seed"]) + 1)

    batch_size = int(cfg["data"]["batch_size"])
    forget_loader = DataLoader(forget_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    forget_name = cfg["data"]["forget"]["class_name"]
    forget_domain = cfg["data"]["forget"]["domain"]
    print(f"Forget : {forget_name} ({forget_domain}) — {len(forget_train)} train samples")
    print(f"Retain : {len(retain_train)} train samples (includes other-domain {forget_name})")

    # ── CELL 2: E0 + DINO subspace weights ────────────────────────────────────
    E0 = EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=int(cfg["model"].get("num_classes", 10)),
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "resnet18")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 2)),
        imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
    )
    E0 = load_pretrained(E0, str(ROOT / cfg["pretrain"]["checkpoint_path"]), device=device)

    print("Loading DINO encoder...")
    enc_model, enc_preprocess, enc_type = load_dino_encoder(device)
    print("DINO loaded.")

    dset_dino = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
        transform=enc_preprocess,
    )
    forget_dino_all, retain_dino_all = split_forget_retain(dset_dino, forget_spec, retain_spec)
    forget_dino_train, _ = train_holdout_split(forget_dino_all, holdout_fraction, seed=int(cfg["seed"]))
    retain_dino_train, _ = train_holdout_split(retain_dino_all, holdout_fraction, seed=int(cfg["seed"]) + 1)

    forget_dino_loader = DataLoader(forget_dino_train, batch_size=batch_size, shuffle=False, num_workers=0)
    retain_dino_loader = DataLoader(retain_dino_train, batch_size=batch_size, shuffle=False, num_workers=0)

    n_components = int(cfg["unlearning"].get("n_pca_components", 20))
    print(f"\nComputing DINOv2 PCA subspace weights (k={n_components})...")
    retain_weights, _, _ = compute_domainnet_subspace_weights(
        model=enc_model,
        forget_loader=forget_dino_loader,
        retain_loader=retain_dino_loader,
        device=device,
        n_components=n_components,
        encoder_type=enc_type,
    )

    # masking flag — identical to the notebook's commented/uncommented line
    forget_cls = int(cfg["data"]["forget"]["class_label"])
    retain_cls = retain_dino_train.base.targets[retain_dino_train.indices]
    retain_weights = retain_weights.clone()
    if args.mask_mode == "cross-domain":
        retain_weights[retain_cls != forget_cls] = 0.0   # isolated: only forget-class other-domain samples
        print("  mask-mode=cross-domain: only forget-class retain samples receive the signal (isolated)")
    else:
        print("  mask-mode=cross-class: all retain samples receive the weighted forget signal (propagation)")
    print(f"  (forget-class retain samples: {int((retain_cls == forget_cls).sum())})")

    # Sharpen the propagation weighting: zero out low-similarity (unrelated) samples,
    # then optionally raise the survivors to a power. This concentrates the forget
    # signal on the most similar classes and spares unrelated ones (higher unrel. retain).
    w_thresh = float(cfg["unlearning"].get("weight_threshold", 0.0))
    w_power = float(cfg["unlearning"].get("weight_power", 1.0))
    n_before = int((retain_weights > 0).sum())
    if w_thresh > 0.0:
        retain_weights[retain_weights < w_thresh] = 0.0
    if w_power != 1.0:
        retain_weights = retain_weights ** w_power
    if w_thresh > 0.0 or w_power != 1.0:
        n_after = int((retain_weights > 0).sum())
        print(f"  sharpen: threshold={w_thresh} power={w_power} -> nonzero weights {n_before} -> {n_after}"
              f"  (mean={float(retain_weights.mean()):.4f} max={float(retain_weights.max()):.4f})")

    # free DINO before training (not needed further)
    del enc_model, enc_preprocess
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── CELL 4: UnlearnConfig + trainable copy + weighted retain loader ───────
    checkpoint_path = str(ROOT / cfg["unlearning"]["checkpoint_path"])
    un_cfg = UnlearnConfig(
        steps=int(cfg["unlearning"]["steps"]),
        lr=float(cfg["unlearning"]["lr"]),
        weight_decay=float(cfg["unlearning"]["weight_decay"]),
        lambda_f=float(cfg["unlearning"]["lambda_f"]),
        lambda_r=float(cfg["unlearning"]["lambda_r"]),
        lambda_m=float(cfg["unlearning"]["lambda_m"]),
        lambda_e=float(cfg["unlearning"]["lambda_e"]),
        lambda_clip=float(cfg["unlearning"].get("lambda_clip", 0.0)),
        n_pca_components=n_components,
        margin=float(cfg["unlearning"]["margin"]),
        log_every=int(cfg["unlearning"]["log_every"]),
        checkpoint_path=checkpoint_path,
    )

    E = deepcopy(E0)
    E.train()
    for p in E.parameters():
        p.requires_grad_(True)
    if hasattr(E, "_backbone") and E._backbone is not None:
        E._set_resnet_trainable_stages(int(cfg["model"].get("finetune_stages", 2)))

    retain_train_weighted = WeightedSubset(retain_train, retain_weights)
    retain_loader = DataLoader(retain_train_weighted, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    trainable = sum(p.numel() for p in E.parameters() if p.requires_grad)
    print(f"\nlambda_clip={un_cfg.lambda_clip}  lambda_r={un_cfg.lambda_r}  steps={un_cfg.steps}  "
          f"k={un_cfg.n_pca_components}  batch_size={batch_size}")
    print(f"Trainable params: {trainable:,}")
    print(f"Checkpoint -> {checkpoint_path}")

    # ── CELL 5: train ─────────────────────────────────────────────────────────
    unlearn(
        E, E0, forget_loader, retain_loader,
        device=device, cfg=un_cfg, logger=logger, tracker=tracker, seed=int(cfg["seed"]),
    )
    if tracker is not None:
        tracker.close()
    print(f"\nDone. Unlearned checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
