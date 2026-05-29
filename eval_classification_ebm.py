from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    p = argparse.ArgumentParser(description="Classification via argmin_y E(x,y) for a label-conditioned EBM.")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--forget-label", type=int, default=None, help="If set, report forget/retain accuracies too.")

    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--y-chunk", type=int, default=10, help="Chunk size for label sweep to control GPU memory.")
    args = p.parse_args()

    root = _project_root()
    sys.path.insert(0, str(root.parent))

    from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
    from ebm_unlearning.src.evaluation.classification import evaluate_classification
    from ebm_unlearning.src.models.ebm import EnergyModel
    from ebm_unlearning.src.training.pretrain import load_pretrained

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dset = load_dataset(
        DatasetSpec(
            name=str(args.dataset),
            data_dir=str(root / args.data_dir),
            train=False,
            download=True,
        )
    )
    loader = DataLoader(
        dset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = EnergyModel(
        in_channels=int(args.in_channels),
        hidden_dim=int(args.hidden_dim),
        num_classes=int(args.num_classes),
        embed_dim=int(args.embed_dim),
    )
    model = load_pretrained(model, str(Path(args.checkpoint).expanduser()), device=device)
    model.eval()

    res = evaluate_classification(
        model,
        loader,
        device=device,
        num_classes=int(args.num_classes),
        forget_label=(int(args.forget_label) if args.forget_label is not None else None),
        y_chunk=int(args.y_chunk),
    )

    print(f"EBM classification (argmin energy) on {args.dataset} test")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {device}")
    print(f"  overall accuracy: {res.overall_accuracy:.4f}")
    if args.forget_label is not None:
        print(f"  forget label: {int(args.forget_label)}")
        print(f"  forget accuracy: {res.forget_accuracy:.4f}")
        print(f"  retain accuracy: {res.retain_accuracy:.4f}")

    print("  per-class accuracy:")
    for c in range(int(args.num_classes)):
        print(f"    {c}: {res.per_class_accuracy[c]:.4f}")

    print("  confusion matrix:")
    print(res.confusion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


