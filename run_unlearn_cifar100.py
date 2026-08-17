"""
Headless CIFAR-100 cross-class unlearning (mirrors the notebook), parameterized by
--finetune-stages so we can test whether freezing more of the backbone reduces the
broad collateral damage. Saves to a DISTINCT checkpoint name (never overwrites).

Recipe (from configs/config.yaml unlearning): steps 2000, lr 1e-4,
lambda_f/r/m/e = 1/10/1/1e-3, lambda_clip 6, n_pca_components 30, margin 5,
DINOv2 subspace weighting, cross-class (all retain samples weighted).

Usage:
    conda run -n myn_again python run_unlearn_cifar100.py \
        --forget-label 11 --forget-name boy --finetune-stages 1 \
        --out outputs/checkpoints/cifar100_boy_stages1.pt
"""
from __future__ import annotations
import argparse, sys
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.training.unlearn import UnlearnConfig, unlearn
from ebm_unlearning.src.data.dataset import IndexedSubset
from ebm_unlearning.src.data.split import train_holdout_split
from ebm_unlearning.src.losses.clip_subspace import (
    compute_clip_subspace_weights, load_dino_encoder, WeightedSubset)
from ebm_unlearning.src.utils.logging import setup_logger
from ebm_unlearning.src.utils.seed import set_seed


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="outputs/checkpoints/ebm_pretrained_cifar100.pt")
    p.add_argument("--out", required=True, help="new checkpoint path (must not overwrite an existing one)")
    p.add_argument("--forget-label", type=int, default=11)
    p.add_argument("--forget-name", default="boy")
    p.add_argument("--finetune-stages", type=int, default=1, help="1=layer4 only, 2=layer3+4, 0=frozen")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lambda-clip", type=float, default=6.0)
    p.add_argument("--n-pca", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = parse()
    if Path(ROOT / a.out).exists():
        raise SystemExit(f"refusing to overwrite existing checkpoint: {a.out}")
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("unlearn_cifar100", log_file=str(ROOT / "outputs" / "logs" / "unlearn_cifar100.log"))

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.CIFAR100(root=str(ROOT / "data"), train=True, download=True, transform=tf)
    raw = train.data  # (N,32,32,3) uint8 for DINO feature extraction
    targets = torch.tensor(train.targets)

    F = a.forget_label
    forget_idx = torch.nonzero(targets == F, as_tuple=False).squeeze(1)
    retain_idx = torch.nonzero(targets != F, as_tuple=False).squeeze(1)
    forget_sub = IndexedSubset(train, forget_idx)
    retain_sub = IndexedSubset(train, retain_idx)
    forget_tr, _ = train_holdout_split(forget_sub, 0.2, seed=a.seed)
    retain_tr, _ = train_holdout_split(retain_sub, 0.2, seed=a.seed + 1)
    print(f"forget {a.forget_name} train={len(forget_tr)}  retain train={len(retain_tr)}")

    # DINOv2 cross-class subspace weights for retain samples (relative to boy)
    print("loading DINOv2 encoder + computing subspace weights...")
    enc, pre, etype = load_dino_encoder(dev)
    weights, _, _ = compute_clip_subspace_weights(
        enc, pre, raw,
        forget_indices=forget_tr.indices.tolist(),
        retain_indices=retain_tr.indices.tolist(),
        device=dev, n_components=a.n_pca, encoder_type=etype)
    del enc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def make(stages):
        return EnergyModel(in_channels=3, hidden_dim=64, num_classes=100, embed_dim=128,
                           backbone="resnet18", finetune_stages=stages, imagenet_pretrained=False)
    E0 = load_pretrained(make(2), str(ROOT / a.pretrained), device=dev)
    E = deepcopy(E0)
    E._set_resnet_trainable_stages(a.finetune_stages)   # <<< the test: freeze more of the backbone
    E.train()
    trainable = sum(p.numel() for p in E.parameters() if p.requires_grad)
    print(f"finetune_stages={a.finetune_stages}  trainable params={trainable:,}")

    retain_weighted = WeightedSubset(retain_tr, weights)
    forget_loader = DataLoader(forget_tr, batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)
    retain_loader = DataLoader(retain_weighted, batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)

    cfg = UnlearnConfig(
        steps=a.steps, lr=1.0e-4, weight_decay=0.0,
        lambda_f=1.0, lambda_r=10.0, lambda_m=1.0, lambda_e=1.0e-3,
        lambda_clip=a.lambda_clip, n_pca_components=a.n_pca, margin=5.0,
        log_every=100, checkpoint_path=str(ROOT / a.out))
    E = unlearn(E, E0, forget_loader, retain_loader, device=dev, cfg=cfg, logger=logger, seed=a.seed)
    print(f"\nDone. Saved -> {a.out}")


if __name__ == "__main__":
    main()
