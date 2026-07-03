"""
LoRA unlearning runner — identical pipeline to run_unlearn_domainnet.py, but the
backbone is frozen and adapted only via low-rank LoRA adapters (src/models/lora.py).

Why: answers "is your unlearning just retraining the features?" — with LoRA the
original backbone weights are 100% frozen; only rank-r adapters (~0.1-1% of params)
plus the small head (proj/label_emb/energy) train. After training, adapters are
MERGED into the weights and a STANDARD EnergyModel checkpoint is saved, so
analyze_similarity_degradation.py and all eval scripts work unchanged.

This file does NOT modify run_unlearn_domainnet.py or any training/eval code.

Usage:
    conda run -n myn_again python run_unlearn_lora_domainnet.py \
        --config configs/config_ours_lora.yaml --mask-mode cross-class \
        --lora-targets layer4 --lora-rank 8 --lora-alpha 16

Note (same caveat as the finetune_stages=0 frozen run): backbone conv/linear
WEIGHTS are frozen (only LoRA adapters move), but BatchNorm running statistics
in the backbone still update in train() mode. The "not retraining" claim refers
to the learnable weights (frozen + rank-r correction), which is what matters.
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
from ebm_unlearning.src.models.lora import (
    inject_lora, merge_lora, lora_trainable_params,
    LAYER4_CONV_TARGETS, LAYER3_CONV_TARGETS, PROJ_TARGET,
)
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.training.unlearn import UnlearnConfig, unlearn
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to a DomainNet config yaml (relative to project root or absolute).")
    p.add_argument("--mask-mode", choices=["cross-class", "cross-domain"], default="cross-class")
    p.add_argument("--tracker", choices=["null", "tensorboard"], default="null")
    p.add_argument("--lora-targets", choices=["layer4", "layer34", "proj", "both"], default="layer4",
                   help="Which modules get LoRA adapters. layer4 = last backbone stage. "
                        "layer34 = layer3+layer4 (more capacity -> stronger propagation). "
                        "proj = the 512->128 head only. both = proj+layer4.")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
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

    forget_cls = int(cfg["data"]["forget"]["class_label"])
    retain_cls = retain_dino_train.base.targets[retain_dino_train.indices]
    retain_weights = retain_weights.clone()
    if args.mask_mode == "cross-domain":
        retain_weights[retain_cls != forget_cls] = 0.0
        print("  mask-mode=cross-domain: only forget-class retain samples receive the signal (isolated)")
    else:
        print("  mask-mode=cross-class: all retain samples receive the weighted forget signal (propagation)")
    print(f"  (forget-class retain samples: {int((retain_cls == forget_cls).sum())})")

    del enc_model, enc_preprocess
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── CELL 4: UnlearnConfig + LoRA-injected trainable copy ──────────────────
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

    # LoRA: freeze backbone, add rank-r adapters on the chosen targets.
    # Head (proj/label_emb/energy) stays fully trainable UNLESS it is itself a target.
    targets = {"layer4": LAYER4_CONV_TARGETS,
               "layer34": LAYER3_CONV_TARGETS + LAYER4_CONV_TARGETS,
               "proj": PROJ_TARGET,
               "both": PROJ_TARGET + LAYER4_CONV_TARGETS}[args.lora_targets]
    inject_lora(E, targets, rank=args.lora_rank, alpha=args.lora_alpha)
    E = E.to(device)

    retain_train_weighted = WeightedSubset(retain_train, retain_weights)
    retain_loader = DataLoader(retain_train_weighted, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    trainable = sum(p.numel() for p in E.parameters() if p.requires_grad)
    print(f"\nLoRA targets={args.lora_targets}  rank={args.lora_rank}  alpha={args.lora_alpha}")
    print(f"  adapter params: {lora_trainable_params(E):,}   |   total trainable (adapters+head): {trainable:,}")
    print(f"lambda_clip={un_cfg.lambda_clip}  lambda_r={un_cfg.lambda_r}  steps={un_cfg.steps}  batch_size={batch_size}")
    print(f"Checkpoint -> {checkpoint_path}")

    # ── CELL 5: train (unlearn() is UNCHANGED — Adam over params; only adapters+head have grad)
    E = unlearn(
        E, E0, forget_loader, retain_loader,
        device=device, cfg=un_cfg, logger=logger, tracker=tracker, seed=int(cfg["seed"]),
    )
    if tracker is not None:
        tracker.close()

    # ── MERGE adapters into weights and RE-SAVE as a STANDARD checkpoint ───────
    # unlearn() already saved a LoRA-state checkpoint at checkpoint_path; overwrite
    # it with the merged plain-EnergyModel state_dict so all evaluators work as-is.
    merge_lora(E)
    torch.save({"model": E.state_dict()}, checkpoint_path)
    print(f"\nDone. Merged (standard) unlearned checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
