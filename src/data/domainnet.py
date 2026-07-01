from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DOMAINS = ["real", "sketch", "clipart", "painting"]

# 10-class DomainNet subset for tiger unlearning experiment
EXPERIMENT_CLASSES = [
    "tiger",    # 0 — explicit forget target
    "lion",     # 1 — primary cross-class (big cat)
    "bear",     # 2 — secondary cross-class (large predator)
    "zebra",    # 3 — mammal, lower similarity
    "dog",      # 4 — animal control
    "horse",    # 5 — animal control
    "truck",    # 6 — unrelated (vehicle)
    "car",      # 7 — unrelated (vehicle)
    "guitar",   # 8 — unrelated (instrument)
    "airplane", # 9 — unrelated (transport)
]


def domainnet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class DomainNetSubset(Dataset):
    """
    Subset of DomainNet with configurable classes and domains.

    Each sample returns (image_tensor, class_label).
    Domain information stored in self.domain_labels for splitting.
    """

    def __init__(
        self,
        root: str,
        classes: List[str] = EXPERIMENT_CLASSES,
        domains: List[str] = DOMAINS,
        transform=None,
    ):
        self.root = Path(root)
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.domains = domains
        self.transform = transform or domainnet_transform()

        self.samples: List[Tuple[str, int, str]] = []  # (path, class_idx, domain)

        for domain in self.domains:
            for cls in self.classes:
                cls_dir = self.root / domain / cls
                if not cls_dir.exists():
                    continue
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                        self.samples.append((str(img_path), self.class_to_idx[cls], domain))

        if not self.samples:
            raise FileNotFoundError(f"No images found under {root} for classes {classes} / domains {domains}")

        print(f"[domainnet] {len(self.samples)} images | {len(self.classes)} classes × {len(self.domains)} domains")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, class_label, _ = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), class_label

    @property
    def targets(self) -> torch.Tensor:
        return torch.tensor([s[1] for s in self.samples], dtype=torch.long)

    @property
    def domain_labels(self) -> torch.Tensor:
        """Integer domain index for each sample. Used for domain-aware splitting."""
        domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}
        return torch.tensor([domain_to_idx[s[2]] for s in self.samples], dtype=torch.long)

    def get_domain_name(self, idx: int) -> str:
        return self.samples[idx][2]
