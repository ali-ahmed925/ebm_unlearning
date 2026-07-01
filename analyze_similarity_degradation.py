"""
Similarity-vs-degradation analysis for the cross-class unlearning experiment.

This is the cherry-pick-proof result: it quantifies the paper's core claim —
"forgetting propagates by visual-semantic similarity" — over EVERY retain class
in the subset, not a hand-picked few. One dot per class, one correlation number.

What it does
------------
1. Loads the subset config (configs/config_domainnet_subset.yaml).
2. Loads the subset pretrained (E0) and unlearned (E) checkpoints.
3. Evaluates per-class accuracy across all domains for both models, then
   computes each retain class's mean accuracy drop (pretrained - unlearned).
4. Pulls each class's DINOv2 similarity-to-tiger from outputs/similarity_subset.json
   (computed BEFORE unlearning — a results-independent x-axis).
5. Reports Spearman rho between similarity and degradation, and writes:
     - outputs/similarity_vs_degradation.png    (scatter + trend)
     - outputs/similarity_vs_degradation.csv     (raw data for the paper)

Run:
    conda run -n myn_again python analyze_similarity_degradation.py

Read-only w.r.t. checkpoints/config; only creates the two output artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.evaluation.classification import predict_argmin_energy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/config_domainnet_subset.yaml")
    p.add_argument("--unlearned", default="outputs/checkpoints/ebm_unlearned_domainnet_subset_tiger_sketch.pt")
    p.add_argument("--similarity-json", default="outputs/similarity_subset.json")
    p.add_argument("--png-out", default="outputs/similarity_vs_degradation.png")
    p.add_argument("--csv-out", default="outputs/similarity_vs_degradation.csv")
    p.add_argument("--y-chunk", type=int, default=5, help="Label-sweep chunk (lower if OOM).")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    num_classes = int(cfg["model"].get("num_classes", 10))
    dn_root = str(ROOT / cfg["data"]["data_dir"])
    classes = cfg["data"]["classes"]                       # config order
    domains = cfg["data"]["domains"]
    forget_name = cfg["data"]["forget"]["class_name"]
    forget_dom = cfg["data"]["forget"]["domain"]

    def make_model() -> EnergyModel:
        return EnergyModel(
            in_channels=int(cfg["model"]["in_channels"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            num_classes=num_classes,
            embed_dim=int(cfg["model"].get("embed_dim", 128)),
            backbone=str(cfg["model"].get("backbone", "resnet18")),
            finetune_stages=int(cfg["model"].get("finetune_stages", 2)),
            imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
        )

    print("Loading models...")
    E0 = load_pretrained(make_model(), str(ROOT / cfg["pretrain"]["checkpoint_path"]), device=device)
    E = load_pretrained(make_model(), str(ROOT / args.unlearned), device=device)
    E0.eval(); E.eval()
    print(f"  Pretrained : {cfg['pretrain']['checkpoint_path']}")
    print(f"  Unlearned  : {args.unlearned}")

    # ── Per-class accuracy averaged over domains, for one model ──────────────────
    def per_class_acc(model) -> dict[str, list[float]]:
        # collect per (class, domain) accuracy, then average over domains per class
        acc_by_class: dict[str, list[float]] = {c: [] for c in classes}
        for domain in domains:
            dset_d = DomainNetSubset(root=dn_root, classes=classes, domains=[domain])
            loader = DataLoader(dset_d, batch_size=args.batch_size, shuffle=False, num_workers=0)
            yt, yp = predict_argmin_energy(model, loader, device=device,
                                           num_classes=num_classes, y_chunk=args.y_chunk)
            for idx, cls in enumerate(dset_d.classes):     # dset_d.classes is SORTED
                mask = yt == idx
                if mask.sum() == 0:
                    continue
                acc_by_class[cls].append(float(np.mean(yp[mask] == idx)))
        return acc_by_class

    print(f"\nEvaluating pretrained over {len(domains)} domains...")
    pre = per_class_acc(E0)
    print(f"Evaluating unlearned over {len(domains)} domains...")
    unl = per_class_acc(E)

    # ── Similarity (computed pre-unlearning, results-independent x-axis) ─────────
    sim_path = ROOT / args.similarity_json
    sim_map: dict[str, float] = {}
    if sim_path.exists():
        data = json.loads(sim_path.read_text())
        sim_map = {c: float(s) for c, s in data.get("full_ranking", [])}
        sim_map[data.get("forget_class", forget_name)] = 1.0   # forget class == self-similar
    else:
        print(f"  [warn] {sim_path} not found — x-axis similarities unavailable.")

    # ── Assemble per-class rows ──────────────────────────────────────────────────
    rows = []
    for cls in classes:
        if not pre[cls] or not unl[cls]:
            continue
        pre_m = float(np.mean(pre[cls]))
        unl_m = float(np.mean(unl[cls]))
        rows.append({
            "class": cls,
            "similarity": sim_map.get(cls, float("nan")),
            "pre_acc": pre_m,
            "unl_acc": unl_m,
            "degradation": pre_m - unl_m,         # fraction in [0,1]
            "is_forget": cls == forget_name,
        })
    rows.sort(key=lambda r: (np.nan_to_num(r["similarity"], nan=-1)), reverse=True)

    # ── Correlation over RETAIN classes only (the propagation claim) ─────────────
    retain = [r for r in rows if not r["is_forget"] and not np.isnan(r["similarity"])]
    sims = np.array([r["similarity"] for r in retain])
    degs = np.array([r["degradation"] for r in retain])
    rho = pear = float("nan")
    if len(retain) >= 3:
        from scipy.stats import spearmanr, pearsonr
        rho = float(spearmanr(sims, degs).statistic)
        pear = float(pearsonr(sims, degs).statistic)

    # ── Report ────────────────────────────────────────────────────────────────
    W = 60
    print("\n" + "=" * W)
    print(f"  SIMILARITY vs DEGRADATION — forget: {forget_name} ({forget_dom})")
    print("=" * W)
    print(f"  {'Class':18}{'sim':>7}{'pre':>8}{'unl':>8}{'drop':>8}")
    print("  " + "-" * (W - 2))
    for r in rows:
        tag = "  <- FORGET" if r["is_forget"] else ""
        print(f"  {r['class']:18}{r['similarity']:>7.3f}{r['pre_acc']:>8.1%}"
              f"{r['unl_acc']:>8.1%}{r['degradation']:>+8.1%}{tag}")
    print("=" * W)
    print(f"  Retain classes: {len(retain)}")
    print(f"  Spearman rho (similarity vs degradation): {rho:+.3f}")
    print(f"  Pearson  r   (similarity vs degradation): {pear:+.3f}")
    print("  (positive rho => forgetting propagates by similarity, as claimed)")

    # ── CSV ─────────────────────────────────────────────────────────────────────
    csv_path = ROOT / args.csv_out
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "similarity", "pre_acc", "unl_acc", "degradation", "is_forget"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[analyze] data -> {csv_path}")

    # ── Scatter plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        rx = sims
        ry = degs * 100.0
        ax.scatter(rx, ry, s=45, c="#1f77b4", alpha=0.8, edgecolors="k", linewidths=0.5, label="retain class")
        for r in retain:
            ax.annotate(r["class"], (r["similarity"], r["degradation"] * 100.0),
                        fontsize=7, xytext=(3, 3), textcoords="offset points", alpha=0.7)
        # forget class marker (similarity 1.0)
        fr = next((r for r in rows if r["is_forget"]), None)
        if fr is not None:
            ax.scatter([fr["similarity"]], [fr["degradation"] * 100.0], s=90, marker="*",
                       c="#d62728", edgecolors="k", linewidths=0.6, label=f"forget ({forget_name})", zorder=5)
        # trend line over retain points
        if len(retain) >= 2:
            b, a = np.polyfit(rx, ry, 1)
            xs = np.linspace(rx.min(), rx.max(), 50)
            ax.plot(xs, b * xs + a, "--", c="gray", lw=1.2,
                    label=f"trend (Spearman ρ={rho:+.2f})")
        ax.set_xlabel("DINOv2 similarity to tiger (computed pre-unlearning)")
        ax.set_ylabel("accuracy drop after unlearning (%)")
        ax.set_title(f"Forgetting propagates by visual-semantic similarity\nforget: {forget_name} ({forget_dom})")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        png_path = ROOT / args.png_out
        fig.savefig(png_path, dpi=150)
        print(f"[analyze] figure -> {png_path}")
    except Exception as e:
        print(f"  [warn] plot skipped: {e}")


if __name__ == "__main__":
    main()
