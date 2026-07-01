from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ebm_unlearning.src.data.dataset import IndexedSubset, get_targets

# CIFAR-100 vehicles_1 superclass hierarchy (superclass_id → fine class ids)
# Computed from: fine_label=8:bicycle, 13:bus, 48:motorcycle, 58:pickup_truck, 90:train
CIFAR100_SUPERCLASS_TO_FINE: dict = {
    0:  [4, 30, 55, 72, 95],   # aquatic_mammals
    1:  [1, 32, 67, 73, 91],   # fish
    2:  [54, 62, 70, 82, 92],  # flowers
    3:  [9, 10, 16, 28, 61],   # food_containers
    4:  [0, 51, 53, 57, 83],   # fruit_and_vegetables
    5:  [22, 39, 40, 86, 87],  # household_electrical_devices
    6:  [5, 20, 25, 84, 94],   # household_furniture
    7:  [6, 7, 14, 18, 24],    # insects
    8:  [3, 42, 43, 88, 97],   # large_carnivores
    9:  [12, 17, 37, 68, 76],  # large_man-made_outdoor_things
    10: [23, 33, 49, 60, 71],  # large_natural_outdoor_scenes
    11: [15, 19, 21, 31, 38],  # large_omnivores_and_herbivores
    12: [34, 63, 64, 66, 75],  # medium_mammals
    13: [26, 45, 77, 79, 99],  # non-insect_invertebrates
    14: [2, 11, 35, 46, 98],   # people
    15: [27, 29, 44, 78, 93],  # reptiles
    16: [36, 50, 65, 74, 80],  # small_mammals
    17: [47, 52, 56, 59, 96],  # trees
    18: [8, 13, 48, 58, 90],   # vehicles_1: bicycle,bus,motorcycle,pickup_truck,train
    19: [41, 69, 81, 85, 89],  # vehicles_2: lawn_mower,rocket,streetcar,tank,tractor
}


@dataclass(frozen=True)
class ForgetSpec:
    mode: Literal["class", "superclass", "class_domain"] = "class"
    class_label: int = 0
    superclass_label: Optional[int] = None   # used when mode="superclass"
    domain: Optional[str] = None              # used when mode="class_domain" (DomainNet)


@dataclass(frozen=True)
class RetainSpec:
    mode: Literal["complement"] = "complement"


def _get_forget_mask(dataset: Dataset, forget: ForgetSpec) -> torch.Tensor:
    targets = get_targets(dataset)
    if forget.mode == "class":
        return targets == int(forget.class_label)
    if forget.mode == "superclass":
        sc = int(forget.superclass_label)
        fine_classes = CIFAR100_SUPERCLASS_TO_FINE.get(sc)
        if fine_classes is None:
            raise ValueError(f"Unknown superclass_label={sc}")
        mask = torch.zeros(len(targets), dtype=torch.bool)
        for fc in fine_classes:
            mask |= (targets == fc)
        return mask
    if forget.mode == "class_domain":
        # DomainNet: forget a specific (class, domain) combination
        # dataset must expose .domain_labels
        if not hasattr(dataset, "domain_labels"):
            raise ValueError("mode='class_domain' requires dataset to expose .domain_labels")
        from ebm_unlearning.src.data.domainnet import DOMAINS
        domain_idx = DOMAINS.index(forget.domain)
        domain_labels = dataset.domain_labels
        class_mask  = targets == int(forget.class_label)
        domain_mask = domain_labels == domain_idx
        return class_mask & domain_mask
    raise ValueError(f"Unsupported forget.mode={forget.mode}")


def split_forget_retain(
    dataset: Dataset, forget: ForgetSpec, retain: RetainSpec
) -> Tuple[IndexedSubset, IndexedSubset]:
    if retain.mode != "complement":
        raise ValueError(f"Unsupported retain.mode={retain.mode}")

    forget_mask = _get_forget_mask(dataset, forget)
    retain_mask = ~forget_mask

    forget_idx = torch.nonzero(forget_mask, as_tuple=False).squeeze(1)
    retain_idx = torch.nonzero(retain_mask, as_tuple=False).squeeze(1)
    return IndexedSubset(dataset, forget_idx), IndexedSubset(dataset, retain_idx)


def train_holdout_split(
    subset: IndexedSubset, holdout_fraction: float, seed: int
) -> Tuple[IndexedSubset, IndexedSubset]:
    if not (0.0 < float(holdout_fraction) < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")

    n = len(subset)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)

    n_holdout = int(round(n * float(holdout_fraction)))
    holdout_local = perm[:n_holdout]
    train_local = perm[n_holdout:]

    train_idx = subset.indices[train_local]
    holdout_idx = subset.indices[holdout_local]
    return IndexedSubset(subset.base, train_idx), IndexedSubset(subset.base, holdout_idx)


