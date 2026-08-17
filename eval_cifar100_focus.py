"""
Standalone CIFAR-100 evaluation for a single forget class, mirroring the notebook
output: RESULTS (forget/retain acc, forgetting rate, model utility), a PER-CLASS
focus table (related + unrelated classes), and EBM ENERGY METRICS (energy gap,
Spearman rho retain-energy preservation).

Usage:
    conda run -n myn_again python eval_cifar100_focus.py \
        --pretrained outputs/checkpoints/ebm_pretrained_cifar100.pt \
        --unlearned  outputs/checkpoints/cifar100_unlearned_boy.pt \
        --forget-label 11 --forget-name boy
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.evaluation.classification import predict_argmin_energy

# focus classes for the people superclass (boy). Related = people; unrelated = mixed.
RELATED = {2: "baby", 35: "girl", 46: "man", 98: "woman"}
UNRELATED = {0: "apple", 48: "motorcycle", 69: "rocket", 84: "table", 86: "telephone", 39: "keyboard"}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="outputs/checkpoints/ebm_pretrained_cifar100.pt")
    p.add_argument("--unlearned", default="outputs/checkpoints/cifar100_unlearned_boy.pt")
    p.add_argument("--forget-label", type=int, default=11)
    p.add_argument("--forget-name", default="boy")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--y-chunk", type=int, default=10)
    return p.parse_args()


def main():
    a = parse()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test = datasets.CIFAR100(root=str(ROOT / "data"), train=False, download=True, transform=tf)
    loader = DataLoader(test, batch_size=a.batch_size, shuffle=False, num_workers=2)

    def make():
        return EnergyModel(in_channels=3, hidden_dim=64, num_classes=100, embed_dim=128,
                           backbone="resnet18", finetune_stages=2, imagenet_pretrained=False)
    E0 = load_pretrained(make(), str(ROOT / a.pretrained), device=dev); E0.eval()
    E = load_pretrained(make(), str(ROOT / a.unlearned), device=dev); E.eval()

    F = a.forget_label
    yt0, yp0 = predict_argmin_energy(E0, loader, device=dev, num_classes=100, y_chunk=a.y_chunk)
    yt, yp = predict_argmin_energy(E, loader, device=dev, num_classes=100, y_chunk=a.y_chunk)

    def acc(yt, yp, c):
        m = yt == c
        return float((yp[m] == c).mean()) if m.sum() else float("nan")

    # energies at the TRUE label, for the energy-gap and preservation metrics
    def e_at_label(model):
        es, ls = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(dev); y = y.to(dev).long()
                es.append(model(x, y).cpu().numpy()); ls.append(y.cpu().numpy())
        return np.concatenate(es), np.concatenate(ls)
    e0, lab = e_at_label(E0)
    e1, _ = e_at_label(E)

    fa0, fa = acc(yt0, yp0, F), acc(yt, yp, F)
    ra0 = float(np.nanmean([acc(yt0, yp0, c) for c in range(100) if c != F]))
    ra = float(np.nanmean([acc(yt, yp, c) for c in range(100) if c != F]))
    fr = (1 - fa / fa0) if fa0 > 0 else float("nan")
    mu = ra / ra0 if ra0 > 0 else float("nan")

    fmask = lab == F; rmask = ~fmask
    gap0 = float(e0[fmask].mean() - e0[rmask].mean())
    gap1 = float(e1[fmask].mean() - e1[rmask].mean())
    rho = float(spearmanr(e0[rmask], e1[rmask]).statistic)

    line = "=" * 60
    print(f"\n{line}\n  RESULTS — CIFAR100 — forget: {a.forget_name}\n{line}")
    print(f"  {'Metric':38}{'Pretrained':>11}{'Unlearned':>11}")
    print(f"  {'Forget accuracy':38}{fa0:>10.1%}{fa:>11.1%}")
    print(f"  {'Retain accuracy':38}{ra0:>10.1%}{ra:>11.1%}")
    print(f"  {'Forgetting rate (higher is better)':38}{'—':>10}{fr:>11.1%}")
    print(f"  {'Model utility   (higher is better)':38}{'—':>10}{mu:>11.1%}")

    print(f"\n{line}\n  PER-CLASS ACCURACY — CIFAR100 — forget: {a.forget_name}\n{line}")
    print(f"  {'Class':16}{'Label':>6}{'Pretrained':>12}{'Unlearned':>11}{'Change':>9}   group")
    focus = [(a.forget_name, F, "FORGET")] + [(n, c, "related") for c, n in RELATED.items()] \
            + [(n, c, "unrelated") for c, n in UNRELATED.items()]
    for name, c, grp in focus:
        p0, p1 = acc(yt0, yp0, c), acc(yt, yp, c)
        tag = " <- FORGET" if grp == "FORGET" else ""
        print(f"  {name:16}{c:>6}{p0:>12.1%}{p1:>11.1%}{(p1-p0):>+9.1%}   {grp}{tag}")
    rel_pre = np.nanmean([acc(yt0, yp0, c) for c in RELATED]); rel_unl = np.nanmean([acc(yt, yp, c) for c in RELATED])
    unr_pre = np.nanmean([acc(yt0, yp0, c) for c in UNRELATED]); unr_unl = np.nanmean([acc(yt, yp, c) for c in UNRELATED])
    print(f"  {'-'*54}")
    print(f"  {'related (mean)':16}{'':>6}{rel_pre:>12.1%}{rel_unl:>11.1%}{(rel_unl-rel_pre):>+9.1%}")
    print(f"  {'unrelated (mean)':16}{'':>6}{unr_pre:>12.1%}{unr_unl:>11.1%}{(unr_unl-unr_pre):>+9.1%}")

    print(f"\n{line}\n  EBM ENERGY METRICS — CIFAR100 — forget: {a.forget_name}\n{line}")
    print(f"  {'Metric':46}{'Pre':>8}{'Unl':>8}")
    print(f"  {'Energy gap  E(forget) - E(retain) [higher=better]':46}{gap0:>+8.3f}{gap1:>+8.3f}")
    print(f"  {'Retain energy preservation (Spearman rho) [1=best]':46}{1.0:>8.4f}{rho:>8.4f}")
    print()


if __name__ == "__main__":
    main()
