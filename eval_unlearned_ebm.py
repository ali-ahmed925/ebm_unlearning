from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _project_root() -> Path:
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
def _evaluate_unlearned(
    model,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    k_neg: int,
    neg_chunk: int,
) -> dict[str, np.ndarray]:
    """
    Unlearned intent:
      E(x, y_true) should be HIGH
      E(x, y_wrong) should be LOW

    We report:
      inv_gap = E_true - mean(E_wrong)   (positive => wrong labels preferred)
    and also:
      gap = mean(E_wrong) - E_true       (negative => wrong labels preferred)
    """
    if k_neg < 1:
        raise ValueError("k_neg must be >= 1")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if neg_chunk < 1:
        raise ValueError("neg_chunk must be >= 1")

    e_true_all: list[np.ndarray] = []
    e_wrong_mean_all: list[np.ndarray] = []
    inv_gap_all: list[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).to(torch.long)
        b = xb.shape[0]

        e_true = model(xb, yb)  # (B,)

        y_true = yb.view(b, 1)
        r = torch.randint(0, num_classes - 1, size=(b, k_neg), device=device, dtype=torch.long)
        y_wrong = r + (r >= y_true).to(torch.long)  # (B, K)

        # Evaluate wrong-label energies in chunks to avoid OOM for large K.
        e_wrong_chunks: list[torch.Tensor] = []
        for j in range(0, k_neg, neg_chunk):
            y_chunk = y_wrong[:, j : j + neg_chunk]  # (B, kc)
            kc = int(y_chunk.shape[1])
            x_rep = xb.unsqueeze(1).expand(-1, kc, *xb.shape[1:]).reshape(b * kc, *xb.shape[1:])
            y_flat = y_chunk.reshape(b * kc)
            e_chunk = model(x_rep, y_flat).reshape(b, kc)  # (B, kc)
            e_wrong_chunks.append(e_chunk)
        e_wrong = torch.cat(e_wrong_chunks, dim=1)  # (B, K)
        e_wrong_mean = e_wrong.mean(dim=1)  # (B,)

        inv_gap = e_true - e_wrong_mean

        e_true_all.append(e_true.detach().float().cpu().numpy())
        e_wrong_mean_all.append(e_wrong_mean.detach().float().cpu().numpy())
        inv_gap_all.append(inv_gap.detach().float().cpu().numpy())

    e_true_np = np.concatenate(e_true_all, axis=0) if e_true_all else np.empty((0,), dtype=np.float32)
    e_wrong_mean_np = (
        np.concatenate(e_wrong_mean_all, axis=0) if e_wrong_mean_all else np.empty((0,), dtype=np.float32)
    )
    inv_gap_np = np.concatenate(inv_gap_all, axis=0) if inv_gap_all else np.empty((0,), dtype=np.float32)

    return {
        "e_true": e_true_np,
        "e_wrong_mean": e_wrong_mean_np,
        "inv_gap": inv_gap_np,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate an UNLEARNED label-conditioned EBM by inverted energy preference (test split)."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (expects {'model': state_dict}).")
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"], help="Dataset to evaluate.")
    p.add_argument("--data-dir", type=str, default="data", help="Dataset directory (relative to ebm_unlearning/).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--filter-label",
        type=int,
        default=None,
        help="If set, evaluate only test samples with y == this label (typically the forget class).",
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
        help="Chunk size for wrong-label evaluation to reduce GPU memory (use smaller if you OOM).",
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

    dset = load_dataset(DatasetSpec(name=str(args.dataset), data_dir=str(root / args.data_dir), train=False, download=True))
    if args.filter_label is not None:
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

    arrays = _evaluate_unlearned(
        model,
        loader,
        device=device,
        num_classes=int(args.num_classes),
        k_neg=int(args.k_neg),
        neg_chunk=int(args.neg_chunk),
    )
    e_true = arrays["e_true"]
    e_wrong_mean = arrays["e_wrong_mean"]
    inv_gap = arrays["inv_gap"]
    gap = -inv_gap  # mean(E_wrong) - E_true

    print(f"EBM unlearning-intent evaluation ({args.dataset} test)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  device: {device}")
    print(f"  num samples: {int(inv_gap.size)}")
    if args.filter_label is not None:
        print(f"  filter label: {int(args.filter_label)}")
    print(f"  K wrong labels: {args.k_neg}")
    print(f"  trim fraction: {args.trim}")
    print(f"  avg true-label energy: {float(np.mean(e_true)):.6f}")
    print(f"  avg wrong-label energy (mean over K): {float(np.mean(e_wrong_mean)):.6f}")

    # Primary metric for unlearning: inv_gap = E_true - E_wrong_mean
    print(f"  inv_gap = E_true - E_wrong_mean (trimmed mean): {_trimmed_mean(inv_gap, float(args.trim)):.6f}")
    print(f"  inv_gap (median): {float(np.median(inv_gap)):.6f}")
    print(f"  % inv_gap < 0 (FAIL): {float(np.mean(inv_gap < 0.0) * 100.0):.2f}%")

    # Also report the original gap for reference.
    print(f"  gap = E_wrong_mean - E_true (trimmed mean): {_trimmed_mean(gap, float(args.trim)):.6f}")
    print(f"  gap (median): {float(np.median(gap)):.6f}")
    print(f"  % gap < 0 (expected for unlearning): {float(np.mean(gap < 0.0) * 100.0):.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


