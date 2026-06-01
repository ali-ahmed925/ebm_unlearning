"""
Check prediction distribution for forgotten classes.
True unlearning = predictions spread uniformly across many classes (entropy high).
Redirected = predictions collapsed to 1-2 specific wrong labels (entropy low).

Run:
    conda run -n myn_again python check_prediction_distribution.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
from ebm_unlearning.src.evaluation.classification import predict_argmin_energy
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained

import pickle
with open(ROOT / "data" / "cifar-100-python" / "meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")
FINE_NAMES = [s.decode() for s in meta[b"fine_label_names"]]

CHECKPOINT   = ROOT / "outputs" / "checkpoints" / "ebm_unlearned_clip_boy_dino.pt"
DATASET      = "cifar100"
NUM_CLASSES  = 100
CHECK_LABELS = [2, 11, 35, 46, 98]   # baby, boy, girl, man, woman

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EnergyModel(
    in_channels=3, hidden_dim=64, num_classes=NUM_CLASSES,
    embed_dim=128, backbone="resnet18", finetune_stages=2,
)
model = load_pretrained(model, str(CHECKPOINT), device=device)

spec = DatasetSpec(name=DATASET, data_dir=str(ROOT / "data"), train=False, download=False)
dset = load_dataset(spec)
loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=2)

print("Running argmin classification on full test set...")
y_true, y_pred = predict_argmin_energy(model, loader, device=device, num_classes=NUM_CLASSES, y_chunk=10)

print()
for label in CHECK_LABELS:
    mask = y_true == label
    preds = y_pred[mask]
    n = len(preds)

    # Top predicted classes
    counts = Counter(preds.tolist())
    top5 = counts.most_common(5)

    # Entropy of prediction distribution (higher = more random)
    probs = np.array([counts.get(c, 0) / n for c in range(NUM_CLASSES)])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(NUM_CLASSES)  # uniform = log(100) ≈ 4.605

    correct = int(np.sum(preds == label))

    print(f"Class {label:2d} ({FINE_NAMES[label]:6s}) | n={n} | correct={correct} ({100*correct/n:.0f}%)")
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} ({100*entropy/max_entropy:.1f}% of max) {'← random ✓' if entropy/max_entropy > 0.8 else '← structured ✗' if entropy/max_entropy < 0.5 else '← mixed'}")
    print(f"  Top-5 predicted labels:")
    for cls_id, cnt in top5:
        marker = " ← CORRECT" if cls_id == label else ""
        print(f"    {cls_id:3d} ({FINE_NAMES[cls_id]:20s}): {cnt:3d}/{n} ({100*cnt/n:.0f}%){marker}")
    print()
