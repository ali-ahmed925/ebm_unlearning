"""
Subspace overlap test: do trucks share the same PCA variation axes as automobiles?

Mean alignment (previous test) proves trucks lean toward automobiles.
This test proves whether trucks VARY the same way automobiles vary —
which is the actual requirement for PCA subspace generalization.

Run:
    conda run -n myn_again python check_similarity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.utils.seed import set_seed

CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat",  4: "deer",
    5: "dog",      6: "frog",       7: "horse", 8: "ship", 9: "truck"
}

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading CLIP...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model = clip_model.to(device).eval()

cifar = CIFAR10(root=str(ROOT / "data"), train=True, transform=clip_preprocess, download=True)
targets = torch.tensor(cifar.targets)


@torch.no_grad()
def get_features(cls_id: int) -> torch.Tensor:
    idx = (targets == cls_id).nonzero(as_tuple=False).squeeze(1).tolist()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(cifar, idx), batch_size=256, shuffle=False, num_workers=2
    )
    vecs = []
    for xb, _ in loader:
        feats = clip_model.encode_image(xb.to(device)).float()
        vecs.append(F.normalize(feats, dim=-1))
    return torch.cat(vecs, dim=0)  # (N, D)


print("Extracting features for all classes...")
features = {cls_id: get_features(cls_id) for cls_id in range(10)}

# Global mean across all classes (removes cone)
global_mean = torch.cat(list(features.values())).mean(0)

# Center each class
centered = {cls_id: features[cls_id] - global_mean.unsqueeze(0) for cls_id in range(10)}

# PCA on automobile (class 1) — get top-k principal directions
print("\nComputing automobile PCA subspace...")
auto_centered = centered[1]
_, _, Vt = torch.linalg.svd(auto_centered, full_matrices=False)  # Vt: (D, D)

print("\n=== Subspace overlap test: explained variance in automobile PCA ===")
print(f"{'Class':12}  {'k=1':>6}  {'k=5':>6}  {'k=10':>6}  {'k=20':>6}  {'k=50':>6}")
print("-" * 55)

for cls_id in [1, 9, 0, 3, 5, 7]:  # auto, truck, airplane, cat, dog, horse
    f = centered[cls_id]
    total_var = (f ** 2).sum(dim=1).mean().item()
    row = f"{CIFAR10_CLASSES[cls_id]:12}"
    for k in [1, 5, 10, 20, 50]:
        Vk = Vt[:k]                          # (k, D)
        proj = f @ Vk.T                      # (N, k)
        explained = (proj ** 2).sum(dim=1).mean().item()
        frac = explained / total_var
        row += f"  {frac:>5.1%}"
    print(row)

print()
print("Interpretation:")
print("  If truck k=10 explained variance ≈ automobile k=10 → PCA subspace generalizes ✓")
print("  If truck k=10 ≈ cat k=10 → subspace is not discriminative ✗")
print()

# Also show cumulative variance explained by automobile's own PCA
print("=== Automobile's own PCA: cumulative variance explained ===")
auto_total = (auto_centered ** 2).sum(dim=1).mean().item()
for k in [1, 5, 10, 20, 50]:
    Vk = Vt[:k]
    proj = auto_centered @ Vk.T
    frac = (proj ** 2).sum(dim=1).mean().item() / auto_total
    print(f"  Top-{k:2d} PCs explain {frac:.1%} of automobile variance")
