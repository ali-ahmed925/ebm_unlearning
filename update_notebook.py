import json

path = '/home/owais/machine unlearning/ebm_unlearning/notebooks/03_unlearning.ipynb'

with open(path, 'r') as f:
    nb = json.load(f)

cell1_src = """from __future__ import annotations
# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — Classification Accuracy + MIA
#  Set these two variables and run. No other cells needed.
# ══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE          = "config.yaml"
UNLEARNED_CHECKPOINT = "outputs/checkpoints/ebm_unlearned_clip.pt"
# ══════════════════════════════════════════════════════════════════════════════
import sys
import numpy as np
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

def _find_root():
    p = Path(".").resolve()
    for _ in range(6):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("project root not found")

ROOT = _find_root()
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.evaluation.classification import predict_argmin_energy
from ebm_unlearning.src.evaluation.metrics import collect_energies, membership_inference_proxy

with open(ROOT / "configs" / CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = cfg["data"]["dataset"]
forget_label = int(cfg["data"]["forget"]["class_label"])
forget_mode  = cfg["data"]["forget"]["mode"]
num_classes  = int(cfg["model"].get("num_classes", 10))
y_chunk      = max(5, num_classes // 20)

def _make_model():
    return EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "conv")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 1)),
        imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
    )

print("Loading models...")
E0 = load_pretrained(_make_model(), str(ROOT / cfg["pretrain"]["checkpoint_path"]), device=device)
E  = load_pretrained(_make_model(), str(ROOT / UNLEARNED_CHECKPOINT), device=device)
E0.eval(); E.eval()
print(f"  Pretrained : {cfg['pretrain']['checkpoint_path']}")
print(f"  Unlearned  : {UNLEARNED_CHECKPOINT}")

if dataset_name == "domainnet":
    dset_train = DomainNetSubset(root=str(ROOT / cfg["data"]["data_dir"]),
                                 classes=cfg["data"]["classes"], domains=cfg["data"]["domains"])
    forget_domain = cfg["data"]["forget"].get("domain", "")
    class_name    = cfg["data"]["forget"].get("class_name", cfg["data"]["classes"][forget_label])
    forget_spec   = ForgetSpec(mode=forget_mode, class_label=forget_label, domain=forget_domain)
    dset_test = dset_train   # DomainNet has no separate test split
else:
    dset_train    = load_dataset(DatasetSpec(name=dataset_name, data_dir=str(ROOT / cfg["data"]["data_dir"]), train=True,  download=True))
    dset_test     = load_dataset(DatasetSpec(name=dataset_name, data_dir=str(ROOT / cfg["data"]["data_dir"]), train=False, download=True))
    class_name    = dset_train.classes[forget_label] if hasattr(dset_train, "classes") else str(forget_label)
    forget_spec   = ForgetSpec(mode=forget_mode, class_label=forget_label)
    forget_domain = None

# MIA uses training splits (train vs holdout) ─────────────────────────────────
forget_all, retain_all = split_forget_retain(dset_train, forget_spec, RetainSpec())
hf   = float(cfg["evaluation"]["holdout_fraction"])
seed = int(cfg["seed"])
forget_train, forget_holdout = train_holdout_split(forget_all, hf, seed=seed)
retain_train, retain_holdout = train_holdout_split(retain_all, hf, seed=seed + 1)
bs = int(cfg["data"]["batch_size"])
fl_tr = DataLoader(forget_train,   batch_size=bs, shuffle=False, num_workers=0)
fl_ho = DataLoader(forget_holdout, batch_size=bs, shuffle=False, num_workers=0)
rl_tr = DataLoader(retain_train,   batch_size=bs, shuffle=False, num_workers=0)
rl_ho = DataLoader(retain_holdout, batch_size=bs, shuffle=False, num_workers=0)

# Classification uses test set ─────────────────────────────────────────────────
forget_test, retain_test = split_forget_retain(dset_test, forget_spec, RetainSpec())
fl_test = DataLoader(forget_test, batch_size=bs, shuffle=False, num_workers=0)
rl_test = DataLoader(retain_test, batch_size=bs, shuffle=False, num_workers=0)
print(f"  Test  forget={len(forget_test)}  retain={len(retain_test)}")
print(f"  Train forget={len(forget_train)} holdout={len(forget_holdout)} (MIA only)")

def _clf(model, loader):
    return predict_argmin_energy(model, loader, device=device, num_classes=num_classes, y_chunk=y_chunk)
def _acc(yt, yp):
    return float(np.mean(yt == yp)) if len(yt) else float("nan")

print("Classifying on test set...")
yt_f, yp_f_pre = _clf(E0, fl_test);  _,    yp_f_unl = _clf(E,  fl_test)
yt_r, yp_r_pre = _clf(E0, rl_test);  _,    yp_r_unl = _clf(E,  rl_test)

fa_pre = _acc(yt_f, yp_f_pre);  fa_unl = _acc(yt_f, yp_f_unl)
ra_pre = _acc(yt_r, yp_r_pre);  ra_unl = _acc(yt_r, yp_r_unl)
fr = (fa_pre - fa_unl) / fa_pre if fa_pre > 0 else float("nan")
mu = ra_unl / ra_pre             if ra_pre > 0 else float("nan")

print("Computing MIA on training splits...")
mia_pre = membership_inference_proxy(collect_energies(E0, fl_tr, device=device), collect_energies(E0, fl_ho, device=device))
mia_unl = membership_inference_proxy(collect_energies(E,  fl_tr, device=device), collect_energies(E,  fl_ho, device=device))
mia_ret = membership_inference_proxy(collect_energies(E,  rl_tr, device=device), collect_energies(E,  rl_ho, device=device))

W = 54
print()
print("=" * W)
print(f"  RESULTS — {dataset_name.upper()} — forget: {class_name}")
print("=" * W)
print(f"  {'Metric':<36} {'Pretrained':>9} {'Unlearned':>9}")
print(f"  {'-' * (W - 2)}")
print(f"  {'Forget accuracy':<36} {fa_pre:>8.1%} {fa_unl:>9.1%}")
print(f"  {'Retain accuracy':<36} {ra_pre:>8.1%} {ra_unl:>9.1%}")
print(f"  {'Forgetting rate  (higher is better)':<36} {'—':>9} {fr:>9.1%}")
print(f"  {'Model utility    (higher is better)':<36} {'—':>9} {mu:>9.1%}")
print(f"  {'-' * (W - 2)}")
print(f"  {'MIA forget pretrained  (higher=overfit)':<36} {mia_pre:>9.4f} {'—':>9}")
print(f"  {'MIA forget unlearned   (0.5=perfect)':<36} {'—':>9} {mia_unl:>9.4f}")
print(f"  {'MIA retain unlearned   (0.5=perfect)':<36} {'—':>9} {mia_ret:>9.4f}")
print("=" * W)

if dataset_name == "domainnet":
    from ebm_unlearning.src.data.domainnet import DOMAINS
    from ebm_unlearning.src.data.dataset import IndexedSubset
    print(f"\n  Cross-domain breakdown — class: {class_name}")
    print(f"  {'Domain':<12} {'Pretrained':>11} {'Unlearned':>11} {'Drop':>8}")
    print(f"  {'-' * 46}")
    for dom in DOMAINS:
        d    = DomainNetSubset(root=str(ROOT / cfg["data"]["data_dir"]),
                               classes=cfg["data"]["classes"], domains=[dom])
        mask = d.targets == forget_label
        if mask.sum() == 0:
            continue
        idx  = torch.nonzero(mask, as_tuple=False).squeeze(1)
        dl   = DataLoader(__import__('ebm_unlearning.src.data.dataset', fromlist=['IndexedSubset']).IndexedSubset(d, idx),
                          batch_size=32, shuffle=False, num_workers=0)
        _, yp0 = _clf(E0, dl)
        _, yp1 = _clf(E,  dl)
        yt_d   = d.targets[idx].numpy()
        a0 = _acc(yt_d, yp0);  a1 = _acc(yt_d, yp1)
        tag = " <- forget" if dom == forget_domain else ""
        print(f"  {dom:<12} {a0:>10.1%} {a1:>11.1%} {a0-a1:>+8.1%}{tag}")"""

cell2_src = """from __future__ import annotations
# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — Per-Class Accuracy Table
#  Set these two variables and run. No other cells needed.
# ══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE          = "config.yaml"
UNLEARNED_CHECKPOINT = "outputs/checkpoints/ebm_unlearned_clip.pt"
# ══════════════════════════════════════════════════════════════════════════════

import sys
import numpy as np
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

def _find_root():
    p = Path(".").resolve()
    for _ in range(6):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("project root not found")

ROOT = _find_root()
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.evaluation.classification import predict_argmin_energy

with open(ROOT / "configs" / CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = cfg["data"]["dataset"]
forget_label = int(cfg["data"]["forget"]["class_label"])
num_classes  = int(cfg["model"].get("num_classes", 10))
y_chunk      = max(5, num_classes // 20)

def _make_model():
    return EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "conv")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 1)),
        imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
    )

print("Loading models...")
E0 = load_pretrained(_make_model(), str(ROOT / cfg["pretrain"]["checkpoint_path"]), device=device)
E  = load_pretrained(_make_model(), str(ROOT / UNLEARNED_CHECKPOINT), device=device)
E0.eval(); E.eval()

# ── Load evaluation dataset ───────────────────────────────────────────────────
if dataset_name == "domainnet":
    dset = DomainNetSubset(
        root=str(ROOT / cfg["data"]["data_dir"]),
        classes=cfg["data"]["classes"],
        domains=cfg["data"]["domains"],
    )
    class_names   = dset.classes
    forget_domain = cfg["data"]["forget"].get("domain", "")
    class_name    = cfg["data"]["forget"].get("class_name", class_names[forget_label])
else:
    # Use test split for clean per-class evaluation
    spec = DatasetSpec(name=dataset_name, data_dir=str(ROOT / cfg["data"]["data_dir"]),
                       train=False, download=True)
    dset = load_dataset(spec)
    class_names = dset.classes if hasattr(dset, "classes") else [str(i) for i in range(num_classes)]
    class_name  = class_names[forget_label]
    forget_domain = None

loader = DataLoader(dset, batch_size=int(cfg["data"]["batch_size"]),
                    shuffle=False, num_workers=0)
print(f"Evaluating on {len(dset)} {'test' if dataset_name != 'domainnet' else 'full'} samples...")

def _clf(model):
    return predict_argmin_energy(model, loader, device=device,
                                 num_classes=num_classes, y_chunk=y_chunk)

print("Classifying with pretrained model...")
yt, yp_pre = _clf(E0)
print("Classifying with unlearned model...")
_,  yp_unl = _clf(E)

# ── Per-class accuracy ────────────────────────────────────────────────────────
def _per_class_acc(yt, yp):
    return {c: float(np.mean(yp[yt == c] == c)) if (yt == c).sum() > 0 else float("nan")
            for c in range(num_classes)}

acc_pre = _per_class_acc(yt, yp_pre)
acc_unl = _per_class_acc(yt, yp_unl)

# ── Classes to show in table ──────────────────────────────────────────────────
# Empty = show all (good for analysis). Set labels to restrict for paper tables.
# Example for rocket on CIFAR-100:
#   forget class + vehicles_2 superclass + vehicles_1 + unrelated controls
FOCUS_CLASSES = [16, 9, 10, 28, 61, 22, 39, 40, 0, 51, 54, 69]
   # e.g. [69, 41, 78, 82, 86, 8, 13, 48, 90, 51, 54, 0, 22]
# ─────────────────────────────────────────────────────────────────────────────

show = set(FOCUS_CLASSES) if FOCUS_CLASSES else set(range(num_classes))
show.add(forget_label)   # always include the forget class

W = 62
print()
print("=" * W)
print(f"  PER-CLASS ACCURACY — {dataset_name.upper()} — forget: {class_name}")
if FOCUS_CLASSES:
    print(f"  (showing {len(show)} selected classes)")
print("=" * W)
print(f"  {'Class':<22} {'Label':>5} {'Pretrained':>11} {'Unlearned':>10} {'Change':>8}  ")
print(f"  {'-' * (W - 2)}")

for c in range(num_classes):
    if c not in show:
        continue
    name = class_names[c] if c < len(class_names) else str(c)
    pre  = acc_pre[c]
    unl  = acc_unl[c]
    chg  = unl - pre
    flag = " <- FORGET" if c == forget_label else ""
    chg_str = f"{chg:>+7.1%}"
    print(f"  {name:<22} {c:>5} {pre:>10.1%} {unl:>10.1%} {chg_str}{flag}")

print(f"  {'-' * (W - 2)}")
overall_pre = float(np.mean(yt == yp_pre))
overall_unl = float(np.mean(yt == yp_unl))
print(f"  {'OVERALL':<22} {'':>5} {overall_pre:>10.1%} {overall_unl:>10.1%} {overall_unl - overall_pre:>+7.1%}")
print("=" * W)"""

cell3_src = """from __future__ import annotations
# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — EBM Energy Metrics
#  Set these two variables and run. No other cells needed.
# ══════════════════════════════════════════════════════════════════════════════
CONFIG_FILE          = "config.yaml"
UNLEARNED_CHECKPOINT = "outputs/checkpoints/ebm_unlearned_clip.pt"
# ══════════════════════════════════════════════════════════════════════════════

import sys
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import torch
import yaml
from torch.utils.data import DataLoader

def _find_root():
    p = Path(".").resolve()
    for _ in range(6):
        if (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("project root not found")

ROOT = _find_root()
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.data.dataset import DatasetSpec, load_dataset
from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.split import ForgetSpec, RetainSpec, split_forget_retain, train_holdout_split
from ebm_unlearning.src.models.ebm import EnergyModel
from ebm_unlearning.src.training.pretrain import load_pretrained
from ebm_unlearning.src.evaluation.metrics import collect_energies

with open(ROOT / "configs" / CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)

device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = cfg["data"]["dataset"]
forget_label = int(cfg["data"]["forget"]["class_label"])
forget_mode  = cfg["data"]["forget"]["mode"]
num_classes  = int(cfg["model"].get("num_classes", 10))

def _make_model():
    return EnergyModel(
        in_channels=int(cfg["model"]["in_channels"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_classes=num_classes,
        embed_dim=int(cfg["model"].get("embed_dim", 128)),
        backbone=str(cfg["model"].get("backbone", "conv")),
        finetune_stages=int(cfg["model"].get("finetune_stages", 1)),
        imagenet_pretrained=bool(cfg["model"].get("imagenet_pretrained", True)),
    )

print("Loading models...")
E0 = load_pretrained(_make_model(), str(ROOT / cfg["pretrain"]["checkpoint_path"]), device=device)
E  = load_pretrained(_make_model(), str(ROOT / UNLEARNED_CHECKPOINT), device=device)
E0.eval(); E.eval()

# ── Dataset and splits ────────────────────────────────────────────────────────
if dataset_name == "domainnet":
    dset = DomainNetSubset(root=str(ROOT / cfg["data"]["data_dir"]),
                           classes=cfg["data"]["classes"], domains=cfg["data"]["domains"])
    class_name  = cfg["data"]["forget"].get("class_name", cfg["data"]["classes"][forget_label])
    forget_spec = ForgetSpec(mode=forget_mode, class_label=forget_label,
                             domain=cfg["data"]["forget"].get("domain", ""))
else:
    dset        = load_dataset(DatasetSpec(name=dataset_name, data_dir=str(ROOT / cfg["data"]["data_dir"]),
                                           train=True, download=True))
    class_name  = dset.classes[forget_label] if hasattr(dset, "classes") else str(forget_label)
    forget_spec = ForgetSpec(mode=forget_mode, class_label=forget_label)

forget_all, retain_all = split_forget_retain(dset, forget_spec, RetainSpec())
hf   = float(cfg["evaluation"]["holdout_fraction"])
seed = int(cfg["seed"])
forget_train, forget_holdout = train_holdout_split(forget_all, hf, seed=seed)
retain_train, retain_holdout = train_holdout_split(retain_all, hf, seed=seed + 1)
bs = int(cfg["data"]["batch_size"])
fl_ho = DataLoader(forget_holdout, batch_size=bs, shuffle=False, num_workers=0)
rl_tr = DataLoader(retain_train,   batch_size=bs, shuffle=False, num_workers=0)
rl_ho = DataLoader(retain_holdout, batch_size=bs, shuffle=False, num_workers=0)

# ── Collect energies ──────────────────────────────────────────────────────────
print("Collecting energies...")
f_ho_e0 = collect_energies(E0, fl_ho, device=device)
f_ho_e  = collect_energies(E,  fl_ho, device=device)
r_ho_e0 = collect_energies(E0, rl_ho, device=device)
r_ho_e  = collect_energies(E,  rl_ho, device=device)
r_tr_e0 = collect_energies(E0, rl_tr, device=device)
r_tr_e  = collect_energies(E,  rl_tr, device=device)

# ── 1. Energy gap  E(forget) - E(retain) ──────────────────────────────────────
eg_pre = float(f_ho_e0.mean() - r_ho_e0.mean())
eg_unl = float(f_ho_e.mean()  - r_ho_e.mean())

# ── 2. Retain energy preservation (Spearman ρ) ────────────────────────────────
# Measures rank-order preservation of retain energies after unlearning.
# ρ=1.0: ordering identical to pretrained (argmin decisions unaffected).
# ρ=0.0: no rank agreement (energy landscape fully reshuffled).
# Spearman is preferred over Pearson here because EBM inference (argmin) depends
# on rank ordering, not absolute values — uniform shifts leave ρ=1.0 correctly.
rho_unl, _ = spearmanr(r_tr_e0, r_tr_e)
rho_pre     = 1.0   # pretrained vs itself — anchors the scale

# ── Print table ───────────────────────────────────────────────────────────────
W = 62
print()
print("=" * W)
print(f"  EBM ENERGY METRICS — {dataset_name.upper()} — forget: {class_name}")
print("=" * W)
print(f"  {'Metric':<46} {'Pre':>6} {'Unl':>6}")
print(f"  {'-' * (W - 2)}")
print(f"  {'Energy gap  E(forget) - E(retain)  [higher=better]':<46} {eg_pre:>+6.3f} {eg_unl:>+6.3f}")
print(f"  {'Retain energy preservation  (Spearman ρ) [1=best]':<46} {rho_pre:>6.4f} {rho_unl:>6.4f}")
print("=" * W)
print()
print(f"  Interpretation:")
print(f"  Energy gap        : {eg_pre:+.3f} → {eg_unl:+.3f}  ({'large separation' if eg_unl > 1.0 else 'moderate separation' if eg_unl > 0 else 'no separation'})")
print(f"  Retain preservation: ρ = {rho_unl:.4f}  ({'strong' if rho_unl > 0.9 else 'moderate' if rho_unl > 0.7 else 'weak'} rank-order preservation of retain energies)")"""

# Add the cells
def make_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in src.split('\n')]
    }

nb['cells'].extend([make_cell(cell1_src), make_cell(cell2_src), make_cell(cell3_src)])

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook successfully updated.')
