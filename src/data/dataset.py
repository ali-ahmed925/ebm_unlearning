from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data_dir: str
    train: bool = True
    download: bool = True


def _mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _cifar10_transform() -> transforms.Compose:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _cifar100_transform() -> transforms.Compose:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_dataset(spec: DatasetSpec) -> Dataset:
    name = spec.name.lower()
    if name == "mnist":
        print(f"[data] loading mnist (train={spec.train}, download={spec.download}) from {spec.data_dir}")
        return datasets.MNIST(
            root=spec.data_dir,
            train=spec.train,
            download=spec.download,
            transform=_mnist_transform(),
        )
    if name in ("cifar10", "cifar-10"):
        print(f"[data] loading cifar10 (train={spec.train}, download={spec.download}) from {spec.data_dir}")
        return datasets.CIFAR10(
            root=spec.data_dir,
            train=spec.train,
            download=spec.download,
            transform=_cifar10_transform(),
        )
    if name in ("cifar100", "cifar-100"):
        print(f"[data] loading cifar100 (train={spec.train}, download={spec.download}) from {spec.data_dir}")
        return datasets.CIFAR100(
            root=spec.data_dir,
            train=spec.train,
            download=spec.download,
            transform=_cifar100_transform(),
        )
    raise ValueError(f"Unsupported dataset: {spec.name}. Use DomainNetSubset directly for domainnet.")


class IndexedSubset(Dataset):
    def __init__(self, base: Dataset, indices: torch.Tensor):
        self.base = base
        self.indices = indices.to(torch.long).cpu()

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = int(self.indices[idx].item())
        x, y = self.base[base_idx]
        return x, int(y)


def get_targets(dataset: Dataset) -> torch.Tensor:
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return t if isinstance(t, torch.Tensor) else torch.tensor(t)
    raise ValueError("Dataset does not expose `targets` attribute.")


def infer_input_shape(dataset: Dataset) -> Tuple[int, int, int]:
    x0, _ = dataset[0]
    if not isinstance(x0, torch.Tensor):
        raise ValueError("Expected tensor samples.")
    if x0.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x0.shape)}")
    c, h, w = x0.shape
    return int(c), int(h), int(w)


