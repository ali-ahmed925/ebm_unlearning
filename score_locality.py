"""
Score checkpoints under the two-sided locality metrics (DTR / SCD).

This is the E1 driver: it needs NO training. It re-scores checkpoints that
already exist on disk and reports, for each, how much the erasure transferred
across rendering domains (DTR, desired) versus how much it leaked into
semantically adjacent retain classes (SCD, hazardous).

Evaluation uses the exact bilinear factorization of E(x, y) (see
src/evaluation/locality.all_class_energies), so scoring all C labels costs one
backbone pass per image rather than C. On the 26-class DomainNet subset that is
a 26x speedup, and it is verified against the direct E(x, y) path at startup.

Usage
-----
DomainNet, one checkpoint:
    conda run -n myn_again python score_locality.py \
        --config configs/config_domainnet_subset.yaml \
        --unlearned outputs/checkpoints/ebm_unlearned_domainnet_subset_tiger_sketch.pt

Several checkpoints in one dataset pass (pretrained features are reused):
    conda run -n myn_again python score_locality.py \
        --config configs/config_domainnet_subset.yaml \
        --unlearned outputs/checkpoints/a.pt outputs/checkpoints/b.pt \
        --out outputs/locality_e1.json

CIFAR-100 (single domain: DTR is n/a, SCD still meaningful):
    conda run -n myn_again python score_locality.py \
        --config configs/config.yaml \
        --unlearned outputs/checkpoints/cifar100_unlearned_boy.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import DatasetSpec, IndexedSubset, load_dataset
from ebm_unlearning.src.data.split import (
    ForgetSpec,
    RetainSpec,
    split_forget_retain,
    train_holdout_split,
)
from ebm_unlearning.src.evaluation.locality import (
    all_class_energies,
    compute_locality,
    format_cell_table,
    format_report,
    per_cell_accuracy,
    verify_against_reference,
)
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained

CIFAR100_PEOPLE = [2, 11, 35, 46, 98]  # baby, boy, girl, man, woman


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="Config yaml describing dataset + model + forget spec.")
    p.add_argument("--unlearned", nargs="+", required=True, help="One or more unlearned checkpoints to score.")
    p.add_argument("--pretrained", default=None,
                   help="Reference checkpoint. Defaults to pretrain.checkpoint_path from the config.")
    p.add_argument("--split", choices=["holdout", "all"], default="holdout",
                   help="holdout (default, unbiased) or all (matches the published tables).")
    p.add_argument("--batch-size", type=int, default=64, help="Eval batch size (forward only).")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers. Defaults to the config value.")
    p.add_argument("--similarity-json", default="outputs/similarity_subset.json",
                   help="Optional DINO similarity ranking, used for the similarity-weighted SCD.")
    p.add_argument("--out", default=None, help="Write all reports to this JSON path.")
    p.add_argument("--full-table", action="store_true", help="Also print the full per-class per-domain grid.")
    p.add_argument("--limit", type=int, default=None, help="Debug: evaluate at most N samples.")
    p.add_argument("--forget-class", type=int, default=None,
                   help="Override data.forget.class_label. Needed when scoring checkpoints whose "
                        "forget target differs from the config's (e.g. the per-class CIFAR-100 sweep).")
    return p.parse_args()


def build_model(cfg: dict, device: torch.device, path: Path) -> EnergyModel:
    m = EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=int(cfg["model"]["num_classes"]),
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "resnet18")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 2)),
        imagenet_pretrained=False,  # weights come from the checkpoint
    )
    return load_pretrained(m, str(path), device=device)


def build_eval_set(cfg: dict, split: str) -> Tuple[object, torch.Tensor, Optional[np.ndarray], List[str], List[str], int, Optional[int]]:
    """
    Returns (base_dataset, eval_indices, domain_idx_or_None, class_names,
             domain_names, forget_class, forget_domain).
    """
    dname = str(cfg["data"].get("dataset", "")).lower()
    fspec_cfg = cfg["data"]["forget"]
    forget_class = int(fspec_cfg["class_label"])

    if dname == "domainnet":
        from ebm_unlearning.src.data.domainnet import DOMAINS, DomainNetSubset

        dset = DomainNetSubset(
            root=str(ROOT / cfg["data"]["data_dir"]),
            classes=cfg["data"]["classes"],
            domains=cfg["data"]["domains"],
        )
        class_names = list(dset.classes)
        domain_names = list(cfg["data"]["domains"])
        forget_domain_name = str(fspec_cfg["domain"])
        forget_domain = DOMAINS.index(forget_domain_name)
        forget_spec = ForgetSpec(mode="class_domain", class_label=forget_class, domain=forget_domain_name)
        # domain_labels indexes into DOMAINS; remap to the config's domain order
        remap = {DOMAINS.index(d): i for i, d in enumerate(domain_names)}
        forget_domain = remap[forget_domain]
        raw_domain = dset.domain_labels.numpy()
        domain_idx_full = np.array([remap.get(int(v), -1) for v in raw_domain])
    else:
        spec = DatasetSpec(name=dname, data_dir=str(ROOT / cfg["data"]["data_dir"]), train=True, download=False)
        dset = load_dataset(spec)
        n_cls = int(cfg["model"]["num_classes"])
        class_names = list(getattr(dset, "classes", [str(i) for i in range(n_cls)]))
        domain_names = ["-"]
        forget_domain = None
        domain_idx_full = None
        forget_spec = ForgetSpec(mode="class", class_label=forget_class)

    if split == "all":
        eval_idx = torch.arange(len(dset), dtype=torch.long)
    else:
        hf = float(cfg["evaluation"]["holdout_fraction"])
        seed = int(cfg["seed"])
        forget_all, retain_all = split_forget_retain(dset, forget_spec, RetainSpec())
        _, forget_ho = train_holdout_split(forget_all, hf, seed=seed)
        _, retain_ho = train_holdout_split(retain_all, hf, seed=seed + 1)
        eval_idx = torch.cat([forget_ho.indices, retain_ho.indices]).sort().values

    domain_idx = None if domain_idx_full is None else domain_idx_full[eval_idx.numpy()]
    return dset, eval_idx, domain_idx, class_names, domain_names, forget_class, forget_domain


def load_similarities(path: Path, class_names: List[str]) -> Optional[np.ndarray]:
    """Map the DINO similarity ranking onto the dataset's class order."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        ranking = data.get("full_ranking")
        if not ranking:
            return None
        lookup: Dict[str, float] = {}
        for entry in ranking:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                lookup[str(entry[0])] = float(entry[1])
            elif isinstance(entry, dict) and "class" in entry:
                lookup[str(entry["class"])] = float(entry.get("similarity", entry.get("score", 0.0)))
        if not lookup:
            return None
        sims = np.array([lookup.get(c, np.nan) for c in class_names], dtype=np.float64)
        matched = int(np.isfinite(sims).sum())
        frac = matched / max(len(class_names), 1)
        if frac < 0.5:
            # The similarity file was built for a different class list (e.g. the DomainNet
            # tiger ranking applied to CIFAR-100). A partial name overlap would silently
            # produce a meaningless SCD_w, so refuse it rather than report a wrong number.
            print(f"  ! similarity file matches only {matched}/{len(class_names)} classes "
                  f"({frac:.0%}) — it was built for a different class list. Skipping SCD_w.")
            return None
        if matched < len(class_names):
            print(f"  ! similarity file covers {matched}/{len(class_names)} classes; "
                  f"SCD_w uses the matched subset only.")
        return sims
    except Exception as e:  # a malformed similarity file must not kill the scoring run
        print(f"  ! could not read similarities from {path}: {e}")
        return None


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())
    if args.forget_class is not None:
        cfg["data"]["forget"]["class_label"] = int(args.forget_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(cfg["model"]["num_classes"])
    workers = args.num_workers if args.num_workers is not None else int(cfg["data"].get("num_workers", 0))

    print(f"[locality] config={cfg_path.name}  device={device}  split={args.split}")
    dset, eval_idx, domain_idx, class_names, domain_names, forget_class, forget_domain = build_eval_set(cfg, args.split)

    if args.limit is not None:
        eval_idx = eval_idx[: args.limit]
        domain_idx = None if domain_idx is None else domain_idx[: args.limit]

    eval_set = IndexedSubset(dset, eval_idx)
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
    print(f"[locality] evaluating {len(eval_set)} samples | {num_classes} classes x {len(domain_names)} domains")
    print(f"[locality] forget target: {class_names[forget_class]}"
          + (f" ({domain_names[forget_domain]})" if forget_domain is not None else ""))

    sims = load_similarities(
        Path(args.similarity_json) if Path(args.similarity_json).is_absolute() else ROOT / args.similarity_json,
        class_names,
    )
    print(f"[locality] similarity weights: {'loaded' if sims is not None else 'unavailable (SCD_w skipped)'}")

    pre_path = Path(args.pretrained) if args.pretrained else Path(cfg["pretrain"]["checkpoint_path"])
    if not pre_path.is_absolute():
        pre_path = ROOT / pre_path

    print(f"\n[locality] reference: {pre_path.name}")
    E0 = build_model(cfg, device, pre_path)
    worst = verify_against_reference(E0, loader, device, num_classes)
    print(f"[locality] factorization check ok (max |diff| = {worst:.2e})")

    e_pre, y_true = all_class_energies(E0, loader, device)
    pred_pre = e_pre.argmin(axis=1)
    acc_pre, counts = per_cell_accuracy(y_true, pred_pre, domain_idx, num_classes, len(domain_names))
    print(f"[locality] reference overall accuracy: {(y_true == pred_pre).mean():.1%}")
    del E0, e_pre
    torch.cuda.empty_cache()

    results = {}
    for ck in args.unlearned:
        ck_path = Path(ck) if Path(ck).is_absolute() else ROOT / ck
        if not ck_path.exists():
            print(f"\n[locality] SKIP missing checkpoint: {ck_path}")
            continue
        print(f"\n[locality] scoring: {ck_path.name}")
        E = build_model(cfg, device, ck_path)
        e_post, y2 = all_class_energies(E, loader, device)
        assert np.array_equal(y_true, y2), "label order changed between passes"
        pred_post = e_post.argmin(axis=1)
        acc_post, _ = per_cell_accuracy(y2, pred_post, domain_idx, num_classes, len(domain_names))

        report = compute_locality(
            acc_pre=acc_pre,
            acc_post=acc_post,
            counts=counts,
            forget_class=forget_class,
            forget_domain=forget_domain,
            class_names=class_names,
            domain_names=domain_names,
            similarities=sims,
        )
        print(format_report(report))
        if args.full_table:
            print(format_cell_table(report))
        results[ck_path.name] = report.to_dict()
        del E, e_post
        torch.cuda.empty_cache()

    if args.out:
        out_path = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": str(cfg_path.name),
            "split": args.split,
            "n_eval": len(eval_set),
            "forget_class": forget_class,
            "forget_domain": forget_domain,
            "class_names": class_names,
            "domain_names": domain_names,
            "reports": results,
        }, indent=2))
        print(f"\n[locality] wrote {out_path}")


if __name__ == "__main__":
    main()
