"""
E4 (matched) — run the SAME erasure method against many forget targets.

The E4 headline is that erasing a person concept damages its semantic
neighbourhood far more than erasing a vehicle or a piece of furniture. Read off
pre-existing checkpoints that comparison rests on n=1 person target, and on the
assumption that every checkpoint was produced with identical settings. This
script removes both problems: one code path, one hyperparameter set, one seed
policy, an arbitrary list of targets.

It mirrors run_unlearn_cifar100.py exactly (same losses, same UnlearnConfig, same
split, same DINOv2 subspace weighting) with one change: DINOv2 features for the
CIFAR-100 train set are extracted ONCE and cached. The features are per-image and
completely independent of which class is being forgotten -- only the forget/retain
indexing into them changes -- so caching is exact, and it turns ~24 min of encoder
time per target into ~24 min total.

Usage
-----
    conda run --no-capture-output -n myn_again python run_cifar100_targets.py \
        --targets people objects --tag matched
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import IndexedSubset
from ebm_unlearning.src.data.split import train_holdout_split
from ebm_unlearning.src.losses.clip_subspace import (
    WeightedSubset,
    _extract_features,
    load_dino_encoder,
    subspace_weights_from_features,
)
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.training.unlearn import UnlearnConfig, unlearn
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed

# CIFAR-100 fine indices. People is superclass 14; the object targets are drawn
# from vehicles_1 (18) and household_furniture (6) as matched controls.
PEOPLE: Dict[str, int] = {"baby": 2, "boy": 11, "girl": 35, "man": 46, "woman": 98}
OBJECTS: Dict[str, int] = {"bicycle": 8, "motorcycle": 48, "pickup_truck": 58,
                           "wardrobe": 94, "couch": 25}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", nargs="+", default=["people", "objects"],
                   help="'people', 'objects', or explicit class names.")
    p.add_argument("--pretrained", default="outputs/checkpoints/ebm_pretrained_cifar100.pt")
    p.add_argument("--tag", default="matched", help="Subdirectory under outputs/checkpoints/.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lambda-clip", type=float, default=6.0)
    p.add_argument("--n-pca", type=int, default=30)
    p.add_argument("--finetune-stages", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--cache-dir", default="outputs/cache")
    p.add_argument("--dino-batch", type=int, default=256)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def resolve_targets(specs: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    known = {**PEOPLE, **OBJECTS}
    for s in specs:
        if s == "people":
            out.update(PEOPLE)
        elif s == "objects":
            out.update(OBJECTS)
        elif s in known:
            out[s] = known[s]
        else:
            raise SystemExit(f"unknown target '{s}' (known: {sorted(known)} or 'people'/'objects')")
    return out


def get_features(raw, device, cache_dir: Path, dino_batch: int) -> torch.Tensor:
    """DINOv2 features for every CIFAR-100 train image, extracted once."""
    cache_path = cache_dir / "dino_cifar100_train.pt"
    if cache_path.exists():
        feats = torch.load(cache_path, map_location="cpu")
        print(f"[targets] loaded cached features {tuple(feats.shape)} from {cache_path.name}")
        return feats

    print(f"[targets] extracting DINOv2 features for {len(raw)} CIFAR-100 images (one time)...")
    enc, pre, etype = load_dino_encoder(device)
    feats = _extract_features(enc, pre, raw, list(range(len(raw))), device,
                              encoder_type=etype, batch_size=dino_batch).cpu()
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feats, cache_path)
    print(f"[targets] cached -> {cache_path} {tuple(feats.shape)}")
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return feats


def main() -> None:
    args = parse_args()
    targets = resolve_targets(args.targets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "outputs" / "checkpoints" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("targets", log_file=str(ROOT / "outputs" / "logs" / f"targets_{args.tag}.log"))

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR100(root=str(ROOT / "data"), train=True, download=False, transform=tf)
    all_targets = torch.tensor(train.targets)

    cache_dir = Path(args.cache_dir) if Path(args.cache_dir).is_absolute() else ROOT / args.cache_dir
    feats_all = get_features(train.data, device, cache_dir, args.dino_batch)

    manifest = {}
    total = len(targets) * len(args.seeds)
    done = 0
    for name, label in sorted(targets.items(), key=lambda kv: kv[1]):
        for seed in args.seeds:
            done += 1
            ck = out_dir / f"{name}_s{seed}.pt"
            if args.skip_existing and ck.exists():
                print(f"[targets] ({done}/{total}) skip existing {ck.name}")
                manifest[ck.name] = {"class_name": name, "forget_label": label, "seed": seed}
                continue

            print(f"\n[targets] ({done}/{total}) forgetting '{name}' (label {label}), seed {seed}")
            set_seed(seed)
            f_idx = torch.nonzero(all_targets == label, as_tuple=False).squeeze(1)
            r_idx = torch.nonzero(all_targets != label, as_tuple=False).squeeze(1)
            f_tr, _ = train_holdout_split(IndexedSubset(train, f_idx), 0.2, seed=seed)
            r_tr, _ = train_holdout_split(IndexedSubset(train, r_idx), 0.2, seed=seed + 1)

            # Same arithmetic as compute_clip_subspace_weights, but indexing the cache.
            w, _, _ = subspace_weights_from_features(
                feats_all[f_tr.indices], feats_all[r_tr.indices], args.n_pca
            )
            print(f"[targets]   weights mean={w.mean():.4f} max={w.max():.4f} "
                  f"%>0.05={(w > 0.05).float().mean():.1%}")

            def make(stages):
                return EnergyModel(in_channels=3, hidden_dim=64, num_classes=100, embed_dim=128,
                                   backbone="resnet18", finetune_stages=stages, imagenet_pretrained=False)
            E0 = load_pretrained(make(2), str(ROOT / args.pretrained), device=device)
            E = deepcopy(E0)
            E._set_resnet_trainable_stages(args.finetune_stages)
            E.train()

            fl = DataLoader(f_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
            rl = DataLoader(WeightedSubset(r_tr, w), batch_size=args.batch_size, shuffle=True,
                            num_workers=2, drop_last=True)

            cfg = UnlearnConfig(
                steps=args.steps, lr=1.0e-4, weight_decay=0.0,
                lambda_f=1.0, lambda_r=10.0, lambda_m=1.0, lambda_e=1.0e-3,
                lambda_clip=args.lambda_clip, n_pca_components=args.n_pca, margin=5.0,
                log_every=200, checkpoint_path=str(ck))
            unlearn(E, E0, fl, rl, device=device, cfg=cfg, logger=logger, seed=seed)

            manifest[ck.name] = {"class_name": name, "forget_label": label, "seed": seed}
            del E, E0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mpath = out_dir / "manifest.json"
    mpath.write_text(json.dumps({
        "steps": args.steps, "lambda_clip": args.lambda_clip, "n_pca": args.n_pca,
        "finetune_stages": args.finetune_stages, "batch_size": args.batch_size,
        "points": manifest,
    }, indent=2))
    print(f"\n[targets] done — {len(manifest)} checkpoints in {out_dir}")
    print(f"[targets] manifest -> {mpath}")


if __name__ == "__main__":
    main()
