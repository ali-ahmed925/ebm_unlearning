"""
Build the per-class DINOv2 similarity-to-forget-class JSON that
analyze_similarity_degradation.py consumes, for an arbitrary forget class over
the fixed 26-class subset. Mirrors the centroid computation of
select_similarity_subset.py but keeps the subset fixed and only changes the
reference (forget) class.

Run:
    conda run -n myn_again python make_similarity_json.py \
        --config configs/config_lion_sketch.yaml --forget-class lion \
        --out outputs/similarity_lion.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from ebm_unlearning.src.data.domainnet import DomainNetSubset
from ebm_unlearning.src.data.dataset import IndexedSubset, get_targets
from ebm_unlearning.src.losses.clip_subspace import load_dino_encoder, _extract_features_from_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--forget-class", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / a.config))
    classes = cfg["data"]["classes"]
    domains = cfg["data"]["domains"]
    root = str(ROOT / cfg["data"]["data_dir"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc, preprocess, etype = load_dino_encoder(dev)
    dset = DomainNetSubset(root=root, classes=classes, domains=domains, transform=preprocess)
    targets = get_targets(dset)
    dom = dset.domain_labels
    g = torch.Generator().manual_seed(a.seed)

    # centroid per class = mean DINOv2 feature over a per-domain sample, L2-normalized
    centroids = {}
    for ci, cname in enumerate(classes):
        idxs = []
        for di in range(len(domains)):
            pool = torch.nonzero((targets == ci) & (dom == di), as_tuple=False).squeeze(1)
            if len(pool) == 0:
                continue
            take = pool[torch.randperm(len(pool), generator=g)[:a.per_domain]]
            idxs.extend(take.tolist())
        if not idxs:
            continue
        loader = DataLoader(IndexedSubset(dset, torch.tensor(idxs)), batch_size=64, shuffle=False, num_workers=2)
        feats = _extract_features_from_loader(enc, loader, dev, encoder_type=etype).cpu().numpy()  # (n, D)
        c = feats.mean(0)
        centroids[cname] = c / (np.linalg.norm(c) + 1e-8)
        print(f"  {cname:18s} centroid from {len(idxs)} imgs")

    fc = a.forget_class
    if fc not in centroids:
        raise SystemExit(f"forget class {fc} not found among centroids")
    fcen = centroids[fc]
    ranking = sorted(
        [(c, float(np.dot(fcen, v))) for c, v in centroids.items() if c != fc],
        key=lambda kv: kv[1], reverse=True)
    out = {"forget_class": fc, "forget_domain": cfg["data"]["forget"]["domain"],
           "domains": domains, "per_domain_images": a.per_domain, "seed": a.seed,
           "full_ranking": ranking}
    Path(ROOT / a.out).write_text(json.dumps(out, indent=2))
    print("top-5 similar to %s: %s" % (fc, ", ".join(f"{c}={s:.3f}" for c, s in ranking[:5])))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
