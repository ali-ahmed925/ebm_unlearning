"""
Comprehensive machine unlearning benchmark for label-conditioned EBMs.

Usage (DomainNet):
    python eval_unlearning_benchmark.py \
        --config configs/config_domainnet.yaml \
        --pretrained outputs/checkpoints/ebm_pretrained_domainnet.pt \
        --unlearned  outputs/checkpoints/ebm_unlearned_domainnet_tiger_sketch.pt

Usage (CIFAR-10):
    python eval_unlearning_benchmark.py \
        --config configs/config.yaml \
        --pretrained outputs/checkpoints/ebm_pretrained.pt \
        --unlearned  outputs/checkpoints/ebm_unlearned.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _load_model(cfg: dict, checkpoint: str, device: torch.device):
    from ebm_unlearning.src.models.ebm import EnergyModel
    from ebm_unlearning.src.training.pretrain import load_pretrained
    m = EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "conv")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 1)),
    )
    return load_pretrained(m, checkpoint, device=device)


@torch.no_grad()
def _collect_energies(E, loader, device) -> np.ndarray:
    E.eval()
    out = []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device).to(torch.long)
        out.append(E(xb, yb).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,))


@torch.no_grad()
def _collect_all_class_energies(E, loader, device, num_classes, y_chunk=10) -> np.ndarray:
    """Returns (N, C) array: E(x, y) for every y for every sample."""
    E.eval()
    all_rows = []
    for xb, _ in loader:
        xb = xb.to(device)
        b = xb.shape[0]
        chunks = []
        for y0 in range(0, num_classes, y_chunk):
            ys = torch.arange(y0, min(y0 + y_chunk, num_classes), device=device, dtype=torch.long)
            x_rep = xb.unsqueeze(1).expand(b, ys.numel(), *xb.shape[1:]).reshape(b * ys.numel(), *xb.shape[1:])
            y_rep = ys.unsqueeze(0).expand(b, -1).reshape(b * ys.numel())
            e = E(x_rep, y_rep).reshape(b, ys.numel())
            chunks.append(e.cpu().numpy())
        all_rows.append(np.concatenate(chunks, axis=1))  # (b, C)
    return np.concatenate(all_rows, axis=0) if all_rows else np.empty((0, num_classes))


@torch.no_grad()
def _classify(E, loader, device, num_classes, y_chunk=10):
    E.eval()
    yt, yp = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        b = xb.shape[0]
        chunks = []
        for y0 in range(0, num_classes, y_chunk):
            ys = torch.arange(y0, min(y0 + y_chunk, num_classes), device=device, dtype=torch.long)
            x_rep = xb.unsqueeze(1).expand(b, ys.numel(), *xb.shape[1:]).reshape(b * ys.numel(), *xb.shape[1:])
            y_rep = ys.unsqueeze(0).expand(b, -1).reshape(b * ys.numel())
            chunks.append(E(x_rep, y_rep).reshape(b, ys.numel()).cpu())
        e_all = torch.cat(chunks, dim=1)
        yp.append(e_all.argmin(dim=1).numpy())
        yt.append(yb.numpy())
    return np.concatenate(yt), np.concatenate(yp)


def _mia_auc(train_e: np.ndarray, holdout_e: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    x = np.concatenate([train_e, holdout_e])
    y = np.concatenate([np.ones(len(train_e)), np.zeros(len(holdout_e))])
    return float(roc_auc_score(y, x))


def _section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--pretrained", required=True)
    p.add_argument("--unlearned",  required=True)
    p.add_argument("--device",     default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--y-chunk",    type=int, default=5)
    args = p.parse_args()

    root = _project_root()
    sys.path.insert(0, str(root.parent))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    num_classes     = int(cfg["model"]["num_classes"])
    holdout_frac    = float(cfg["evaluation"]["holdout_fraction"])
    seed            = int(cfg["seed"])
    dataset_name    = cfg["data"]["dataset"]
    forget_cls      = int(cfg["data"]["forget"]["class_label"])
    forget_domain   = cfg["data"]["forget"].get("domain", None)
    forget_cls_name = cfg["data"]["forget"].get("class_name", str(forget_cls))

    print(f"\nEBM Unlearning Benchmark")
    print(f"  Dataset   : {dataset_name}")
    print(f"  Forget    : {forget_cls_name}" + (f" / {forget_domain}" if forget_domain else ""))
    print(f"  Device    : {device}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Unlearned : {args.unlearned}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\nLoading models...")
    E0 = _load_model(cfg, args.pretrained, device)
    E  = _load_model(cfg, args.unlearned,  device)

    # ── Build dataset and splits ───────────────────────────────────────────────
    from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split

    if dataset_name == "domainnet":
        from ebm_unlearning.src.data.domainnet import DomainNetSubset
        dset = DomainNetSubset(
            root=str(root / cfg["data"]["data_dir"]),
            classes=cfg["data"]["classes"],
            domains=cfg["data"]["domains"],
        )
    else:
        from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
        dset = load_dataset(DatasetSpec(
            name=dataset_name,
            data_dir=str(root / cfg["data"]["data_dir"]),
            train=False, download=True,
        ))

    forget_mode = cfg["data"]["forget"].get("mode", "class")
    forget_spec = ForgetSpec(
        mode=forget_mode,
        class_label=forget_cls,
        domain=forget_domain,
    )
    forget_all, retain_all = split_forget_retain(dset, forget_spec, RetainSpec())
    forget_train, forget_holdout = train_holdout_split(forget_all, holdout_frac, seed=seed)
    retain_train, retain_holdout = train_holdout_split(retain_all, holdout_frac, seed=seed + 1)

    def _loader(subset, shuffle=False):
        return DataLoader(subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    fl_tr = _loader(forget_train)
    fl_ho = _loader(forget_holdout)
    rl_tr = _loader(retain_train)
    rl_ho = _loader(retain_holdout)

    print(f"  Forget train={len(forget_train)}  holdout={len(forget_holdout)}")
    print(f"  Retain train={len(retain_train)}  holdout={len(retain_holdout)}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Classification Accuracy
    # ══════════════════════════════════════════════════════════════════════════
    _section("1. Classification Accuracy")

    print("  [1/8] classifying forget-train (pretrained)...")
    yt_f_pre_tr, yp_f_pre_tr = _classify(E0, fl_tr, device, num_classes, args.y_chunk)
    print("  [2/8] classifying forget-holdout (pretrained)...")
    yt_f_pre_ho, yp_f_pre_ho = _classify(E0, fl_ho, device, num_classes, args.y_chunk)
    print("  [3/8] classifying forget-train (unlearned)...")
    yt_f_unl_tr, yp_f_unl_tr = _classify(E,  fl_tr, device, num_classes, args.y_chunk)
    print("  [4/8] classifying forget-holdout (unlearned)...")
    yt_f_unl_ho, yp_f_unl_ho = _classify(E,  fl_ho, device, num_classes, args.y_chunk)
    yt_f_pre = np.concatenate([yt_f_pre_tr, yt_f_pre_ho])
    yp_f_pre = np.concatenate([yp_f_pre_tr, yp_f_pre_ho])
    yt_f_unl = np.concatenate([yt_f_unl_tr, yt_f_unl_ho])
    yp_f_unl = np.concatenate([yp_f_unl_tr, yp_f_unl_ho])

    print("  [5/8] classifying retain-train (pretrained)...")
    yt_r_pre_tr, yp_r_pre_tr = _classify(E0, rl_tr, device, num_classes, args.y_chunk)
    print("  [6/8] classifying retain-holdout (pretrained)...")
    yt_r_pre_ho, yp_r_pre_ho = _classify(E0, rl_ho, device, num_classes, args.y_chunk)
    print("  [7/8] classifying retain-train (unlearned)...")
    yt_r_unl_tr, yp_r_unl_tr = _classify(E,  rl_tr, device, num_classes, args.y_chunk)
    print("  [8/8] classifying retain-holdout (unlearned)...")
    yt_r_unl_ho, yp_r_unl_ho = _classify(E,  rl_ho, device, num_classes, args.y_chunk)
    yt_r_pre = np.concatenate([yt_r_pre_tr, yt_r_pre_ho])
    yp_r_pre = np.concatenate([yp_r_pre_tr, yp_r_pre_ho])
    yt_r_unl = np.concatenate([yt_r_unl_tr, yt_r_unl_ho])
    yp_r_unl = np.concatenate([yp_r_unl_tr, yp_r_unl_ho])

    fa_pre = float(np.mean(yt_f_pre == yp_f_pre))
    fa_unl = float(np.mean(yt_f_unl == yp_f_unl))
    ra_pre = float(np.mean(yt_r_pre == yp_r_pre))
    ra_unl = float(np.mean(yt_r_unl == yp_r_unl))

    forgetting_rate = (fa_pre - fa_unl) / fa_pre if fa_pre > 0 else float("nan")
    model_utility   = ra_unl / ra_pre if ra_pre > 0 else float("nan")

    print(f"  {'Metric':<35} {'Pretrained':>12} {'Unlearned':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Forget accuracy':35} {fa_pre:>11.1%} {fa_unl:>11.1%}")
    print(f"  {'Retain accuracy':35} {ra_pre:>11.1%} {ra_unl:>11.1%}")
    print(f"  {'Forgetting rate (↑ better)':35} {'—':>12} {forgetting_rate:>11.1%}")
    print(f"  {'Model utility   (↑ better)':35} {'—':>12} {model_utility:>11.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Membership Inference Attack
    # ══════════════════════════════════════════════════════════════════════════
    _section("2. Membership Inference Attack (MIA)")
    print("  Score = E(x, y_true). AUC near 0.5 = perfect forgetting.")
    print()

    print("  collecting energies (6 passes)...")
    print("  [1/6] forget-train, pretrained..."); f_tr_e0 = _collect_energies(E0, fl_tr, device)
    print("  [2/6] forget-holdout, pretrained..."); f_ho_e0 = _collect_energies(E0, fl_ho, device)
    print("  [3/6] forget-train, unlearned..."); f_tr_e  = _collect_energies(E,  fl_tr, device)
    print("  [4/6] forget-holdout, unlearned..."); f_ho_e  = _collect_energies(E,  fl_ho, device)
    print("  [5/6] retain-train, unlearned..."); r_tr_e  = _collect_energies(E,  rl_tr, device)
    print("  [6/6] retain-holdout, unlearned..."); r_ho_e  = _collect_energies(E,  rl_ho, device)

    mia_pre  = _mia_auc(f_tr_e0, f_ho_e0)
    mia_unl  = _mia_auc(f_tr_e,  f_ho_e)
    mia_ret  = _mia_auc(r_tr_e,  r_ho_e)

    print(f"  {'Metric':<45} {'AUC':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Forget MIA — pretrained (expect ~1.0)':45} {mia_pre:>7.4f}")
    print(f"  {'Forget MIA — unlearned  (↓ better, 0.5=perfect)':45} {mia_unl:>7.4f}")
    print(f"  {'Retain MIA — unlearned  (expect ~0.5)':45} {mia_ret:>7.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — EBM Energy Metrics
    # ══════════════════════════════════════════════════════════════════════════
    _section("3. EBM Energy Metrics  (EBM-specific)")

    print("  collecting retain-train energies (pretrained)...")
    r_tr_e0 = _collect_energies(E0, rl_tr, device)

    forgetting_score = float(f_ho_e.mean() - f_ho_e0.mean())   # how much forget energy rose
    retention_score  = float(np.mean((r_tr_e - r_tr_e0) ** 2)) # MSE drift of retain energies
    energy_gap_pre   = float(f_ho_e0.mean() - r_ho_e.mean())   # forget-retain gap before  (reuse r_ho)
    r_ho_e0          = _collect_energies(E0, rl_ho, device)
    energy_gap_pre   = float(f_ho_e0.mean() - r_ho_e0.mean())
    energy_gap_unl   = float(f_ho_e.mean()  - r_ho_e.mean())

    print(f"\n  Energy gap = mean E(forget,correct) − mean E(retain,correct)")
    print(f"  Positive & large = forget energy is high relative to retain.")
    print()
    print(f"  {'Metric':<45} {'Value':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Energy gap — pretrained':45} {energy_gap_pre:>+10.4f}")
    print(f"  {'Energy gap — unlearned  (↑ better)':45} {energy_gap_unl:>+10.4f}")
    print(f"  {'Forgetting score  (↑ better, Δ forget energy)':45} {forgetting_score:>+10.4f}")
    print(f"  {'Retention score   (↓ better, retain MSE drift)':45} {retention_score:>10.6f}")

    # Energy rank & argmax rate — EBM-specific
    print(f"\n  --- Energy Rank (EBM-specific) ---")
    print(f"  For each forget image, rank of correct label's energy among all {num_classes} classes.")
    print(f"  Rank {num_classes}/{num_classes} = correct label has HIGHEST energy = model won't predict it.")
    print()

    print("  computing all-class energies for forget holdout (unlearned)...")
    forget_all_e  = _collect_all_class_energies(E,  fl_ho, device, num_classes, args.y_chunk)
    print("  computing all-class energies for forget holdout (pretrained)...")
    forget_all_e0 = _collect_all_class_energies(E0, fl_ho, device, num_classes, args.y_chunk)

    def _energy_rank_stats(all_e, correct_label):
        # rank 1 = lowest energy (most likely), rank C = highest energy (least likely)
        ranks = []
        for row in all_e:
            sorted_idx = np.argsort(row)  # ascending
            rank = int(np.where(sorted_idx == correct_label)[0][0]) + 1  # 1-indexed
            ranks.append(rank)
        ranks = np.array(ranks)
        mean_rank      = float(ranks.mean())
        argmax_rate    = float(np.mean(ranks == num_classes))  # fraction where it's the worst
        return mean_rank, argmax_rate

    rank_pre, argmax_pre = _energy_rank_stats(forget_all_e0, forget_cls)
    rank_unl, argmax_unl = _energy_rank_stats(forget_all_e,  forget_cls)

    print(f"  {'Metric':<45} {'Pretrained':>12} {'Unlearned':>12}")
    print(f"  {'-'*70}")
    print(f"  {'Mean energy rank of forget class (↑ better)':45} {rank_pre:>11.2f} {rank_unl:>11.2f}")
    print(f"  {'% images: forget class is argmax  (↑ better)':45} {argmax_pre:>11.1%} {argmax_unl:>11.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Cross-Domain Generalization (DomainNet only)
    # ══════════════════════════════════════════════════════════════════════════
    if dataset_name == "domainnet":
        _section("4. Cross-Domain Forgetting Generalization")
        print("  classifying per domain (4 domains × 2 models)...")
        from ebm_unlearning.src.data.domainnet import DomainNetSubset, DOMAINS
        from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain

        print(f"  {'Domain':<12} {'Pre FA':>8} {'Unl FA':>8} {'Drop':>8} {'Generalized?':>14}")
        print(f"  {'-'*55}")

        for domain in DOMAINS:
            print(f"    domain: {domain}...")
            d_dset = DomainNetSubset(
                root=str(root / cfg["data"]["data_dir"]),
                classes=cfg["data"]["classes"],
                domains=[domain],
            )
            from ebm_unlearning.src.data.split import _get_forget_mask
            from ebm_unlearning.src.data.dataset import IndexedSubset
            import torch as _t
            targets = d_dset.targets
            cls_mask = targets == forget_cls
            if cls_mask.sum() == 0:
                continue
            idx = _t.nonzero(cls_mask, as_tuple=False).squeeze(1)
            d_loader = _loader(IndexedSubset(d_dset, idx))

            yt_d_pre, yp_d_pre = _classify(E0, d_loader, device, num_classes, args.y_chunk)
            yt_d_unl, yp_d_unl = _classify(E,  d_loader, device, num_classes, args.y_chunk)
            fa_d_pre = float(np.mean(yt_d_pre == yp_d_pre))
            fa_d_unl = float(np.mean(yt_d_unl == yp_d_unl))
            drop     = fa_d_pre - fa_d_unl
            is_forget_domain = (domain == forget_domain)
            tag = " ← target" if is_forget_domain else ("✓ generalized" if drop > 0.5 else "")
            print(f"  {domain:<12} {fa_d_pre:>7.1%} {fa_d_unl:>8.1%} {drop:>+7.1%}  {tag}")

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    _section("Summary")
    print(f"  Forget accuracy (holdout+train):  {fa_pre:.1%} → {fa_unl:.1%}  (Δ {fa_pre-fa_unl:+.1%})")
    print(f"  Retain accuracy:                  {ra_pre:.1%} → {ra_unl:.1%}  (Δ {ra_pre-ra_unl:+.1%})")
    print(f"  MIA AUC (forget, unlearned):      {mia_unl:.4f}  (0.5 = perfect)")
    print(f"  Energy gap (unlearned):           {energy_gap_unl:+.4f}")
    print(f"  Argmax rate (unlearned):          {argmax_unl:.1%}  (100% = fully forgotten energetically)")
    print()


if __name__ == "__main__":
    main()
