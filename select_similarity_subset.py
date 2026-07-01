"""
Select a similarity-spanning DomainNet class subset for the cross-class
unlearning experiment — by a rule defined BEFORE any unlearning is run.

Why this exists
---------------
To demonstrate that forgetting propagates by visual-semantic similarity we need
retain classes that span the full similarity spectrum to the forget class
(tiger): a few very similar, several mid, many unrelated. Selecting that subset
by DINOv2 similarity — computed on the *pretrained* features, independent of any
unlearning result — is a principled, results-independent criterion. It is NOT
cherry-picking: every tier is sampled by similarity rank, not by how well a
class happens to forget.

What it does
------------
1. Loads the frozen DINOv2 encoder (same one the method uses for weighting).
2. For every DomainNet class, samples a few images per domain, extracts DINO
   features, and builds a per-class centroid (pooled across all 4 domains).
3. Ranks all classes by cosine similarity of their centroid to tiger's.
4. Selects a ~N-class spectrum-spanning subset (high / mid / low tiers) by a
   deterministic rule.
5. Writes:
     - outputs/similarity_subset.json   (full ranking + selection, for the paper)
     - configs/config_domainnet_subset.yaml  (ready to pretrain; DIFFERENT
       checkpoint names so existing checkpoints are never overwritten)

Run:
    python select_similarity_subset.py                 # default 25-class subset
    python select_similarity_subset.py --subset-size 30 --per-domain 12

This script is read-only w.r.t. existing code/data/checkpoints. It only creates
two NEW files (the json artifact and the subset config).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from ebm_unlearning.src.losses.clip_subspace import load_dino_encoder
from ebm_unlearning.src.utils.seed import set_seed

# Domains used by the experiment (matches config_domainnet.yaml).
DOMAINS = ["real", "sketch", "clipart", "painting"]
FORGET_CLASS = "tiger"
FORGET_DOMAIN = "sketch"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/domainnet/DomainNet",
                   help="DomainNet root (contains real/ sketch/ clipart/ painting/).")
    p.add_argument("--subset-size", type=int, default=26,
                   help="Total classes in the subset (includes tiger). Default 26 = tiger + 4 high + 6 mid + 15 low.")
    p.add_argument("--per-domain", type=int, default=10,
                   help="Images sampled per class per domain for the centroid.")
    p.add_argument("--n-high", type=int, default=4, help="High-similarity classes to keep (excl. tiger).")
    p.add_argument("--n-mid", type=int, default=6, help="Mid-similarity classes to keep.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config-out", default="configs/config_domainnet_subset.yaml")
    p.add_argument("--json-out", default="outputs/similarity_subset.json")
    p.add_argument("--base-config", default="configs/config_domainnet.yaml",
                   help="Config to copy hyperparameters from.")
    return p.parse_args()


def list_classes(data_root: Path) -> list[str]:
    """Classes present in ALL used domains (intersection), so every class is comparable."""
    per_domain = []
    for d in DOMAINS:
        dom_dir = data_root / d
        if not dom_dir.exists():
            raise FileNotFoundError(f"Domain directory missing: {dom_dir}")
        per_domain.append({p.name for p in dom_dir.iterdir() if p.is_dir()})
    common = set.intersection(*per_domain)
    return sorted(common)


def sample_paths(data_root: Path, cls: str, per_domain: int, g: torch.Generator) -> list[Path]:
    """Up to `per_domain` image paths per domain for a class (deterministic given seed)."""
    paths: list[Path] = []
    for d in DOMAINS:
        cls_dir = data_root / d / cls
        if not cls_dir.exists():
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not imgs:
            continue
        if len(imgs) > per_domain:
            idx = torch.randperm(len(imgs), generator=g)[:per_domain].tolist()
            imgs = [imgs[i] for i in idx]
        paths.extend(imgs)
    return paths


@torch.no_grad()
def class_centroid(model, preprocess, enc_type, paths, device, batch_size) -> torch.Tensor | None:
    """Mean L2-normalized DINO feature over a class's sampled images."""
    if not paths:
        return None
    feats = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        imgs = []
        for pth in batch_paths:
            try:
                imgs.append(preprocess(Image.open(pth).convert("RGB")))
            except Exception:
                continue
        if not imgs:
            continue
        xb = torch.stack(imgs).to(device)
        out = model.encode_image(xb).float() if enc_type == "clip" else model(xb).float()
        feats.append(F.normalize(out, dim=-1))
    if not feats:
        return None
    return torch.cat(feats, dim=0).mean(dim=0)  # (D,)


def select_subset(ranked: list[tuple[str, float]], size: int, n_high: int, n_mid: int) -> dict:
    """
    Spectrum-spanning selection from the similarity ranking (tiger excluded from `ranked`).
      - high: top n_high most similar
      - low:  bottom n_low least similar
      - mid:  evenly spaced across the remaining middle band
    Deterministic; tiger is always added back as the forget target.
    """
    n_low = max(0, size - 1 - n_high - n_mid)  # -1 for tiger
    high = ranked[:n_high]
    low = ranked[-n_low:] if n_low > 0 else []
    middle_band = ranked[n_high: len(ranked) - n_low] if n_low > 0 else ranked[n_high:]
    if n_mid > 0 and middle_band:
        step = max(1, len(middle_band) // n_mid)
        mid = [middle_band[i] for i in range(0, len(middle_band), step)][:n_mid]
    else:
        mid = []
    return {"high": high, "mid": mid, "low": low}


def write_subset_config(base_cfg_path: Path, out_path: Path, classes_sorted: list[str]) -> dict:
    """
    Copy the base config, swap in the subset classes + correct forget index +
    num_classes, and rename ALL checkpoints so existing files are never touched.
    Minimal text edits (no yaml dependency) to preserve comments/formatting.
    """
    text = base_cfg_path.read_text()
    lines = text.splitlines()

    # tiger's label is its index in the SORTED subset (the loader sorts classes).
    tiger_idx = classes_sorted.index(FORGET_CLASS)
    num_classes = len(classes_sorted)
    classes_inline = "[" + ", ".join(classes_sorted) + "]"

    out_lines = []
    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("classes:"):
            out_lines.append(f"{indent}classes: {classes_inline}")
        elif stripped.startswith("class_label:"):
            out_lines.append(f"{indent}class_label: {tiger_idx}        # {FORGET_CLASS} index in sorted(classes) — auto-computed")
        elif stripped.startswith("num_classes:"):
            out_lines.append(f"{indent}num_classes: {num_classes}          # subset size (auto)")
        elif stripped.startswith("checkpoint_path:") and "ebm_pretrained_domainnet" in stripped:
            out_lines.append(f"{indent}checkpoint_path: outputs/checkpoints/ebm_pretrained_domainnet_subset.pt")
        elif stripped.startswith("checkpoint_path:") and "ebm_unlearned_domainnet" in stripped:
            out_lines.append(f"{indent}checkpoint_path: outputs/checkpoints/ebm_unlearned_domainnet_subset_tiger_sketch.pt")
        else:
            out_lines.append(line)

    out_path.write_text("\n".join(out_lines) + "\n")
    return {"tiger_idx": tiger_idx, "num_classes": num_classes}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = (ROOT / args.data_dir).resolve()
    classes = list_classes(data_root)
    print(f"[subset] {len(classes)} classes common to all {len(DOMAINS)} domains")
    if FORGET_CLASS not in classes:
        raise ValueError(f"Forget class '{FORGET_CLASS}' not present in all domains.")

    print("[subset] loading DINOv2 encoder...")
    model, preprocess, enc_type = load_dino_encoder(device)

    print(f"[subset] extracting centroids ({args.per_domain} imgs/domain/class)...")
    centroids: dict[str, torch.Tensor] = {}
    for i, cls in enumerate(classes):
        paths = sample_paths(data_root, cls, args.per_domain, g)
        c = class_centroid(model, preprocess, enc_type, paths, device, args.batch_size)
        if c is not None:
            centroids[cls] = c
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(classes)} classes done")

    if FORGET_CLASS not in centroids:
        raise RuntimeError("Could not build tiger centroid (no usable images).")

    tiger_vec = F.normalize(centroids[FORGET_CLASS], dim=0)
    ranking = []
    for cls, vec in centroids.items():
        if cls == FORGET_CLASS:
            continue
        sim = float(F.cosine_similarity(tiger_vec, F.normalize(vec, dim=0), dim=0))
        ranking.append((cls, sim))
    ranking.sort(key=lambda t: t[1], reverse=True)

    sel = select_subset(ranking, args.subset_size, args.n_high, args.n_mid)
    chosen = [FORGET_CLASS] + [c for c, _ in sel["high"] + sel["mid"] + sel["low"]]
    chosen_sorted = sorted(chosen)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  SELECTED SUBSET ({len(chosen)} classes) — similarity to {FORGET_CLASS}")
    print("=" * 60)
    print(f"  {'Class':18}{'cos_sim':>10}   tier")
    print("  " + "-" * 44)
    print(f"  {FORGET_CLASS:18}{1.000:>10.3f}   FORGET")
    for tier in ("high", "mid", "low"):
        for cls, sim in sel[tier]:
            print(f"  {cls:18}{sim:>10.3f}   {tier}")
    print("=" * 60)
    print(f"\n  Full ranking range: {ranking[0][1]:.3f} ({ranking[0][0]}) "
          f"... {ranking[-1][1]:.3f} ({ranking[-1][0]})")

    # ── Artifacts ───────────────────────────────────────────────────────────────
    json_out = (ROOT / args.json_out).resolve()
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps({
        "forget_class": FORGET_CLASS,
        "forget_domain": FORGET_DOMAIN,
        "domains": DOMAINS,
        "per_domain_images": args.per_domain,
        "seed": args.seed,
        "selection_rule": {"subset_size": args.subset_size, "n_high": args.n_high,
                            "n_mid": args.n_mid, "n_low": args.subset_size - 1 - args.n_high - args.n_mid},
        "selected_sorted": chosen_sorted,
        "selected_by_tier": {t: sel[t] for t in ("high", "mid", "low")},
        "full_ranking": ranking,
    }, indent=2))
    print(f"\n[subset] ranking + selection -> {json_out}")

    cfg_out = (ROOT / args.config_out).resolve()
    base_cfg = (ROOT / args.base_config).resolve()
    info = write_subset_config(base_cfg, cfg_out, chosen_sorted)
    print(f"[subset] subset config       -> {cfg_out}")
    print(f"         classes={info['num_classes']}  tiger label (sorted idx)={info['tiger_idx']}")
    print(f"         pretrain ckpt -> outputs/checkpoints/ebm_pretrained_domainnet_subset.pt")
    print(f"         unlearn  ckpt -> outputs/checkpoints/ebm_unlearned_domainnet_subset_tiger_sketch.pt")
    print("\nNext: pretrain with the subset config (point notebook 06 / your pretrain")
    print("entrypoint at configs/config_domainnet_subset.yaml). Existing checkpoints untouched.")


if __name__ == "__main__":
    main()
