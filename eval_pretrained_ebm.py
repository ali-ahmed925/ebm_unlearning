from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _project_root() -> Path:
    # This file lives at: <root>/ebm_unlearning/eval_pretrained_ebm.py
    return Path(__file__).resolve().parent


def _trimmed_mean(x: np.ndarray, trim: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    trim = float(trim)
    if trim <= 0.0:
        return float(x.mean())
    if trim >= 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    xs = np.sort(x)
    k = int(np.floor(trim * xs.size))
    if 2 * k >= xs.size:
        return float(np.median(xs))
    return float(xs[k : xs.size - k].mean())


@torch.no_grad()
def _evaluate(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    k_neg: int,
    neg_chunk: int,
) -> dict[str, float]:
    if k_neg < 1:
        raise ValueError("k_neg must be >= 1")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if neg_chunk < 1:
        raise ValueError("neg_chunk must be >= 1")

    e_pos_all: list[np.ndarray] = []
    e_neg_all: list[np.ndarray] = []
    gap_all: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).to(torch.long)
        b = xb.shape[0]

        e_pos = model(xb, yb)  # (B,)

        # Sample K wrong labels (with replacement), enforce y_neg != y_true by shifting.
        y_true = yb.view(b, 1)
        r = torch.randint(0, num_classes - 1, size=(b, k_neg), device=device, dtype=torch.long)
        y_neg = r + (r >= y_true).to(torch.long)  # (B, K)

        # Evaluate negatives in chunks to avoid OOM for large K.
        e_neg_chunks: list[torch.Tensor] = []
        for j in range(0, k_neg, neg_chunk):
            y_chunk = y_neg[:, j : j + neg_chunk]  # (B, kc)
            kc = int(y_chunk.shape[1])
            x_rep = xb.unsqueeze(1).expand(-1, kc, *xb.shape[1:]).reshape(b * kc, *xb.shape[1:])
            y_flat = y_chunk.reshape(b * kc)
            e_chunk = model(x_rep, y_flat).reshape(b, kc)  # (B, kc)
            e_neg_chunks.append(e_chunk)
        e_neg = torch.cat(e_neg_chunks, dim=1)  # (B, K)

        e_neg_mean = e_neg.mean(dim=1)  # (B,)
        gap = (e_neg_mean - e_pos)  # (B,)

        e_pos_all.append(e_pos.detach().float().cpu().numpy())
        e_neg_all.append(e_neg_mean.detach().float().cpu().numpy())
        gap_all.append(gap.detach().float().cpu().numpy())

    e_pos_np = np.concatenate(e_pos_all, axis=0) if e_pos_all else np.empty((0,), dtype=np.float32)
    e_neg_np = np.concatenate(e_neg_all, axis=0) if e_neg_all else np.empty((0,), dtype=np.float32)
    gap_np = np.concatenate(gap_all, axis=0) if gap_all else np.empty((0,), dtype=np.float32)

    return {
        "avg_pos_energy": float(np.mean(e_pos_np)) if e_pos_np.size else float("nan"),
        "avg_neg_energy": float(np.mean(e_neg_np)) if e_neg_np.size else float("nan"),
        "median_gap": float(np.median(gap_np)) if gap_np.size else float("nan"),
        "pct_gap_lt_0": float(np.mean(gap_np < 0.0) * 100.0) if gap_np.size else float("nan"),
        "_gap_array": gap_np,  # for trimmed mean post-processing
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate a label-conditioned EBM by energy gaps (test split).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (expects {'model': state_dict}).")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100", "domainnet"], help="Dataset to evaluate.")
    p.add_argument("--data-dir", type=str, default="data", help="Dataset directory (relative to ebm_unlearning/).")
    p.add_argument("--domain", type=str, default=None, help="DomainNet domain filter (real/sketch/clipart/painting).")
    p.add_argument("--filter-class-name", type=str, default=None, help="DomainNet class name filter (e.g. tiger).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--filter-label",
        type=int,
        default=None,
        help="If set, evaluate only test samples with y == this label (e.g., forget class).",
    )

    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--backbone", type=str, default="conv", choices=["conv", "resnet18"])
    p.add_argument("--finetune-stages", type=int, default=1)

    p.add_argument("--k-neg", type=int, default=5, help="Number of wrong labels per sample.")
    p.add_argument(
        "--neg-chunk",
        type=int,
        default=10,
        help="Chunk size for negative-label evaluation to reduce GPU memory (use smaller if you OOM).",
    )
    p.add_argument("--trim", type=float, default=0.1, help="Trim fraction for trimmed mean of gaps.")
    p.add_argument("--seed", type=int, default=0, help="Seed for negative label sampling (ensures reproducible eval).")
    args = p.parse_args()
    torch.manual_seed(int(args.seed))

    root = _project_root()
    sys.path.insert(0, str(root.parent))

    from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
    from ebm_unlearning.src.models.ebm import EnergyModel
    from ebm_unlearning.src.training.pretrain import load_pretrained

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dataset == "domainnet":
        from ebm_unlearning.src.data.domainnet import DomainNetSubset, EXPERIMENT_CLASSES
        domains = [args.domain] if args.domain else ["real", "sketch", "clipart", "painting"]
        dset = DomainNetSubset(root=str(root / args.data_dir), classes=EXPERIMENT_CLASSES, domains=domains)
        filter_cls = dset.classes.index(args.filter_class_name) if args.filter_class_name else args.filter_label
        if filter_cls is not None:
            idx = torch.nonzero(dset.targets == int(filter_cls), as_tuple=False).squeeze(1).tolist()
            dset = Subset(dset, idx)
    else:
        dset = load_dataset(DatasetSpec(name=str(args.dataset), data_dir=str(root / args.data_dir), train=False, download=True))
        if args.filter_label is not None:
            if not hasattr(dset, "targets"):
                raise ValueError("Dataset does not expose `targets`; cannot filter by label.")
            targets = dset.targets
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            mask = targets.to(torch.long) == int(args.filter_label)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1).tolist()
            dset = Subset(dset, idx)
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
        backbone=str(args.backbone),
        finetune_stages=int(args.finetune_stages),
    )
    model = load_pretrained(model, str(Path(args.checkpoint).expanduser()), device=device)
    model = model.to(device)
    model.eval()

    out = _evaluate(
        model,
        loader,
        device=device,
        num_classes=int(args.num_classes),
        k_neg=int(args.k_neg),
        neg_chunk=int(args.neg_chunk),
    )
    gaps = out.pop("_gap_array")
    out["trimmed_mean_gap"] = _trimmed_mean(gaps, float(args.trim))

    print(f"EBM energy gap evaluation ({args.dataset} test)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {device}")
    print(f"  num samples: {int(gaps.size)}")
    if args.filter_label is not None:
        print(f"  filter label: {int(args.filter_label)}")
    print(f"  K wrong labels: {args.k_neg}")
    print(f"  trim fraction: {args.trim}")
    print(f"  avg pos energy: {out['avg_pos_energy']:.6f}")
    print(f"  avg neg energy: {out['avg_neg_energy']:.6f}")
    print(f"  gap (trimmed mean): {out['trimmed_mean_gap']:.6f}")
    print(f"  gap (median): {out['median_gap']:.6f}")
    print(f"  % gap < 0: {out['pct_gap_lt_0']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


