"""
Figure: isolated unlearning lifts only the forget-class energy (CIFAR-10).
Computes mean energy at the true label for the forget class and the retain set,
before (pretrained) and after unlearning, for all 10 classes, then plots a
clustered grouped bar chart (4 bars per class).

Run:
    conda run -n myn_again python make_fig_energy_cifar10.py
Outputs:
    outputs/cifar10_energies.json   (raw numbers)
    outputs/cifar10_energy_fig.png  (figure)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained

NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
PRETRAINED = "outputs/checkpoints/ebm_pretrained.pt"


def make_model():
    return EnergyModel(in_channels=3, hidden_dim=64, num_classes=10, embed_dim=128,
                       backbone="resnet18", finetune_stages=2, imagenet_pretrained=False)


def energies_at_label(model, loader, dev):
    es, ls = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev); y = y.to(dev).long()
            es.append(model(x, y).cpu().numpy()); ls.append(y.cpu().numpy())
    return np.concatenate(es), np.concatenate(ls)


def compute():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test = datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=tf)
    loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=2)

    E0 = load_pretrained(make_model(), str(ROOT / PRETRAINED), device=dev); E0.eval()
    e0, lab = energies_at_label(E0, loader, dev)

    rows = []
    for c, name in enumerate(NAMES):
        E = load_pretrained(make_model(), str(ROOT / f"outputs/checkpoints/cifar10_unlearned_{name}.pt"), device=dev)
        E.eval()
        e1, _ = energies_at_label(E, loader, dev)
        fm = lab == c; rm = ~fm
        rows.append(dict(cls=name,
                         f_pre=float(e0[fm].mean()), f_post=float(e1[fm].mean()),
                         r_pre=float(e0[rm].mean()), r_post=float(e1[rm].mean())))
        print(f"{name:11s} forget {rows[-1]['f_pre']:6.2f} -> {rows[-1]['f_post']:6.2f}   "
              f"retain {rows[-1]['r_pre']:6.2f} -> {rows[-1]['r_post']:6.2f}")
    json.dump(rows, open(ROOT / "outputs/cifar10_energies.json", "w"), indent=2)
    return rows


def plot(rows):
    names = [r["cls"] for r in rows]
    fpre = [r["f_pre"] for r in rows]; fpost = [r["f_post"] for r in rows]
    rpre = [r["r_pre"] for r in rows]; rpost = [r["r_post"] for r in rows]
    INK = "#22303C"; GRID = "#E9EEF2"
    C_FB, C_FA, C_RB, C_RA = "#F3C3A7", "#E4572E", "#AFD6CF", "#227C9D"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK})
    x = np.arange(len(names)); w = 0.20
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 1.5*w, fpre,  w, color=C_FB, label="Forget · before", zorder=3)
    ax.bar(x - 0.5*w, fpost, w, color=C_FA, label="Forget · after",  zorder=3)
    ax.bar(x + 0.5*w, rpre,  w, color=C_RB, label="Retain · before", zorder=3)
    ax.bar(x + 1.5*w, rpost, w, color=C_RA, label="Retain · after",  zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("mean energy at true label", fontsize=12.5)
    ax.axhline(0, color="#C7D0D8", lw=1.2, zorder=2)
    ax.set_ylim(-3.5, 10.5); ax.set_yticks([-2, 0, 2, 4, 6, 8, 10])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#C7D0D8")
    ax.yaxis.grid(True, color=GRID, lw=1.1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.11), frameon=False,
              fontsize=11, handlelength=1.2, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/cifar10_energy_fig.png", dpi=185, bbox_inches="tight", facecolor="white")
    print("saved outputs/cifar10_energy_fig.png")


if __name__ == "__main__":
    plot(compute())
