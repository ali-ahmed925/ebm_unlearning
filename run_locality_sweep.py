"""
E2 — sweep the erasure-strength knob and trace the two-sided locality frontier.

This is the clean premise test. E1 compared heterogeneous methods, which
confounds "entangled axes" with "some methods just change the model more".
Here we vary ONE parameter (lambda_clip, optionally n_pca) inside one method and
ask whether style-invariance (DTR) and semantic collateral damage (SCD) still
rise together.

Cost note: lambda_clip does not affect the DINOv2 subspace weights at all, and
n_pca only affects the cheap SVD step. So the encoder pass over ~22k images runs
ONCE and is cached to disk; every sweep point reuses it. Without this the sweep
would spend more time in DINOv2 than in training.

The split seed stays fixed at cfg['seed'] so that every sweep point is evaluated
on the identical holdout set; only the unlearning seed varies across repeats.
That isolates method variance from split variance.

Usage
-----
    conda run --no-capture-output -n myn_again python run_locality_sweep.py \
        --config configs/config_domainnet_subset.yaml \
        --lambda-clip 0 0.5 1 3 6 12 --seeds 0 1 2 \
        --tag sweep

Then score and plot:
    python score_locality.py --config configs/config_domainnet_subset.yaml \
        --unlearned outputs/checkpoints/sweep/*.pt --out outputs/locality_e2.json
    python analyze_locality.py outputs/locality_e2.json --fig outputs/frontier_e2.png
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split
from ebm_unlearning.src.losses.clip_subspace import (
    WeightedSubset,
    _extract_features_from_loader,
    load_dino_encoder,
    style_invariant_weights_from_features,
    style_subspace_from_features,
    subspace_weights_from_features,
)
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.training.unlearn import UnlearnConfig, unlearn
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--lambda-clip", nargs="+", type=float, default=[0.0, 0.5, 1.0, 3.0, 6.0, 12.0],
                   help="Erasure-strength knob to sweep.")
    p.add_argument("--n-pca", nargs="+", type=int, default=None,
                   help="Optional n_pca values (default: the config value only).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Unlearning seeds. The data split seed is always cfg['seed'].")
    p.add_argument("--steps", nargs="+", type=int, default=None,
                   help="One or more values for unlearning.steps. Duration is the parameter that "
                        "actually drives target forgetting, so this is the erasure-strength knob.")
    p.add_argument("--mask-mode", choices=["cross-class", "cross-domain"], default="cross-class")
    p.add_argument("--weighting", choices=["subspace", "style-deflated"], default="subspace",
                   help="subspace: the stock forget-set PCA weighting. style-deflated (E5): project "
                        "out a rendering-style subspace first, so weights track concept identity "
                        "rather than rendering style.")
    p.add_argument("--style-components", type=int, default=10,
                   help="Dimension of the style subspace (style-deflated only).")
    p.add_argument("--style-include-forget-class", action="store_true",
                   help="Allow the forget class to inform the style subspace. Off by default: the "
                        "style estimate should need no supervision about the erased concept.")
    p.add_argument("--tag", default="sweep", help="Subdirectory under outputs/checkpoints/.")
    p.add_argument("--finetune-stages", type=int, default=None,
                   help="Override model.finetune_stages. 0 = head only (no backbone drift), "
                        "1 = layer4. Used to separate collateral damage caused by backbone "
                        "interference from damage caused by head reallocation.")
    p.add_argument("--cache-dir", default="outputs/cache", help="Where DINOv2 features are cached.")
    p.add_argument("--dino-batch", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override data.batch_size. The configs are pinned to 8 for a 6 GB card; "
                        "the forget loss expands to batch x (num_classes-1) images at 224px, so on "
                        "a 24 GB card 32 is comfortable and much faster.")
    p.add_argument("--skip-existing", action="store_true", help="Do not recompute sweep points already on disk.")
    return p.parse_args()


def cache_key(cfg: dict, holdout_fraction: float) -> str:
    """Identity of the feature cache: anything that changes which images get encoded."""
    payload = json.dumps({
        "classes": sorted(cfg["data"]["classes"]),
        "domains": list(cfg["data"]["domains"]),
        "forget": {k: cfg["data"]["forget"][k] for k in ("class_label", "domain") if k in cfg["data"]["forget"]},
        "holdout_fraction": holdout_fraction,
        "seed": int(cfg["seed"]),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def get_features(cfg: dict, holdout_fraction: float, forget_spec: ForgetSpec,
                 device: torch.device, cache_dir: Path, dino_batch: int):
    """Extract (or load) DINOv2 features for the forget/retain TRAIN splits."""
    key = cache_key(cfg, holdout_fraction)
    cache_path = cache_dir / f"dino_{key}.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        print(f"[sweep] loaded cached DINOv2 features from {cache_path.name} "
              f"(forget={tuple(blob['forget'].shape)}, retain={tuple(blob['retain'].shape)})")
        return blob["forget"], blob["retain"]

    print("[sweep] no cache — extracting DINOv2 features (one time)...")
    enc_model, enc_preprocess, enc_type = load_dino_encoder(device)
    dset = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
        transform=enc_preprocess,
    )
    f_all, r_all = split_forget_retain(dset, forget_spec, RetainSpec())
    f_train, _ = train_holdout_split(f_all, holdout_fraction, seed=int(cfg["seed"]))
    r_train, _ = train_holdout_split(r_all, holdout_fraction, seed=int(cfg["seed"]) + 1)

    fl = DataLoader(f_train, batch_size=dino_batch, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    rl = DataLoader(r_train, batch_size=dino_batch, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 0)))
    ff = _extract_features_from_loader(enc_model, fl, device, enc_type).cpu()
    rf = _extract_features_from_loader(enc_model, rl, device, enc_type).cpu()

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"forget": ff, "retain": rf, "key": key}, cache_path)
    print(f"[sweep] cached -> {cache_path}")

    del enc_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ff, rf


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    holdout_fraction = float(cfg["evaluation"]["holdout_fraction"])
    split_seed = int(cfg["seed"])
    batch_size = int(cfg["data"]["batch_size"])
    n_pca_values = args.n_pca if args.n_pca else [int(cfg["unlearning"].get("n_pca_components", 5))]
    steps_values = args.steps if args.steps else [int(cfg["unlearning"]["steps"])]
    if args.finetune_stages is not None:
        cfg["model"]["finetune_stages"] = int(args.finetune_stages)
        print(f"[sweep] finetune_stages overridden -> {args.finetune_stages}")
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = int(args.batch_size)
        batch_size = int(args.batch_size)   # the local was already read from cfg above
        print(f"[sweep] batch_size overridden -> {args.batch_size}")

    out_dir = ROOT / "outputs" / "checkpoints" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("sweep", log_file=str(ROOT / "outputs" / "logs" / f"sweep_{args.tag}.log"))

    forget_spec = ForgetSpec(
        mode="class_domain",
        class_label=int(cfg["data"]["forget"]["class_label"]),
        domain=str(cfg["data"]["forget"]["domain"]),
    )

    # ── data (pixel space, for training) ─────────────────────────────────────
    set_seed(split_seed)
    dset = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
    )
    f_all, r_all = split_forget_retain(dset, forget_spec, RetainSpec())
    forget_train, _ = train_holdout_split(f_all, holdout_fraction, seed=split_seed)
    retain_train, _ = train_holdout_split(r_all, holdout_fraction, seed=split_seed + 1)
    print(f"[sweep] forget_train={len(forget_train)}  retain_train={len(retain_train)}")

    # ── DINOv2 features, once ─────────────────────────────────────────────────
    cache_dir = Path(args.cache_dir) if Path(args.cache_dir).is_absolute() else ROOT / args.cache_dir
    forget_feat, retain_feat = get_features(cfg, holdout_fraction, forget_spec, device, cache_dir, args.dino_batch)

    retain_cls = retain_train.base.targets[retain_train.indices]
    retain_dom = retain_train.base.domain_labels[retain_train.indices]
    forget_cls = int(cfg["data"]["forget"]["class_label"])

    # E5: estimate the rendering-style subspace from the RETAIN pool only. The forget
    # cell never enters it, and by default neither does the forget class in its other
    # domains -- so the style estimate uses no supervision about the erased concept.
    style_subspace = None
    if args.weighting == "style-deflated":
        style_subspace = style_subspace_from_features(
            retain_feat, retain_cls, retain_dom,
            n_components=args.style_components,
            exclude_class=None if args.style_include_forget_class else forget_cls,
        )
        print(f"[sweep] style subspace: {tuple(style_subspace.shape)} "
              f"(forget class {'included' if args.style_include_forget_class else 'excluded'})")

    pre_path = ROOT / cfg["pretrain"]["checkpoint_path"]
    manifest = {}
    total = len(steps_values) * len(args.lambda_clip) * len(n_pca_values) * len(args.seeds)
    done = 0

    for n_pca in n_pca_values:
        if style_subspace is not None:
            base_w, _ = style_invariant_weights_from_features(
                forget_feat, retain_feat, style_subspace, n_pca
            )
        else:
            base_w, _, _ = subspace_weights_from_features(forget_feat, retain_feat, n_pca)
        if args.mask_mode == "cross-domain":
            base_w = base_w.clone()
            base_w[retain_cls != forget_cls] = 0.0
        # Diagnostic: does the weighting actually track the concept across domains?
        same_cls = retain_cls == forget_cls
        if bool(same_cls.any()):
            print(f"[sweep]   w on forget-class/other-domains = {base_w[same_cls].mean():.4f}  "
                  f"vs other classes = {base_w[~same_cls].mean():.4f}  "
                  f"(ratio {float(base_w[same_cls].mean() / base_w[~same_cls].mean().clamp(min=1e-8)):.2f}x)")
        print(f"\n[sweep] n_pca={n_pca}  weights mean={base_w.mean():.4f} max={base_w.max():.4f} "
              f"%>0.05={(base_w > 0.05).float().mean():.1%}")

        for steps in steps_values:
          for lam in args.lambda_clip:
            for seed in args.seeds:
                done += 1
                # steps MUST be in the name: without it a duration sweep silently
                # overwrites itself, since every other field would be identical.
                name = f"k{n_pca}_lam{lam:g}_st{steps}_s{seed}.pt"
                ck = out_dir / name
                if args.skip_existing and ck.exists():
                    print(f"[sweep] ({done}/{total}) skip existing {name}")
                    manifest[name] = {"lambda_clip": lam, "n_pca": n_pca, "seed": seed, "steps": steps}
                    continue

                print(f"\n[sweep] ({done}/{total}) steps={steps}  lambda_clip={lam:g}  "
                      f"n_pca={n_pca}  seed={seed}")
                # Seed the global RNG with the *run* seed, not the split seed, so that
                # loader shuffling and negative sampling genuinely vary across repeats.
                # The forget/retain split itself is unaffected: train_holdout_split takes
                # an explicit seed and uses its own Generator, and was computed above.
                set_seed(seed)
                E0 = EnergyModel(
                    in_channels=int(cfg["model"]["in_channels"]),
                    hidden_dim=int(cfg["model"]["hidden_dim"]),
                    num_classes=int(cfg["model"]["num_classes"]),
                    embed_dim=int(cfg["model"].get("embed_dim", 128)),
                    backbone=str(cfg["model"].get("backbone", "resnet18")),
                    finetune_stages=int(cfg["model"].get("finetune_stages", 1)),
                    imagenet_pretrained=False,
                )
                E0 = load_pretrained(E0, str(pre_path), device=device)

                E = deepcopy(E0)
                E.train()
                for p in E.parameters():
                    p.requires_grad_(True)
                if getattr(E, "_backbone", None) is not None:
                    E._set_resnet_trainable_stages(int(cfg["model"].get("finetune_stages", 1)))

                un_cfg = UnlearnConfig(
                    steps=steps,
                    lr=float(cfg["unlearning"]["lr"]),
                    weight_decay=float(cfg["unlearning"]["weight_decay"]),
                    lambda_f=float(cfg["unlearning"]["lambda_f"]),
                    lambda_r=float(cfg["unlearning"]["lambda_r"]),
                    lambda_m=float(cfg["unlearning"]["lambda_m"]),
                    lambda_e=float(cfg["unlearning"]["lambda_e"]),
                    lambda_clip=float(lam),
                    n_pca_components=n_pca,
                    margin=float(cfg["unlearning"]["margin"]),
                    log_every=int(cfg["unlearning"]["log_every"]),
                    checkpoint_path=str(ck),
                )

                forget_loader = DataLoader(forget_train, batch_size=batch_size, shuffle=True,
                                           num_workers=0, drop_last=True)
                retain_loader = DataLoader(WeightedSubset(retain_train, base_w), batch_size=batch_size,
                                           shuffle=True, num_workers=0, drop_last=True)

                unlearn(E, E0, forget_loader, retain_loader, device=device, cfg=un_cfg,
                        logger=logger, tracker=None, seed=seed)

                manifest[name] = {"lambda_clip": lam, "n_pca": n_pca, "seed": seed, "steps": steps}
                del E, E0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps({"config": cfg_path.name, "mask_mode": args.mask_mode,
                                 "split_seed": split_seed, "points": manifest}, indent=2))
    print(f"\n[sweep] done — {len(manifest)} checkpoints in {out_dir}")
    print(f"[sweep] manifest -> {mpath}")


if __name__ == "__main__":
    main()
