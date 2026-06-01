from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class EnergyModel(nn.Module):
    """Label-conditioned EBM: E(x, y). Outputs scalar energy per (image, label) pair."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_classes: int = 10,
        embed_dim: int = 128,
        backbone: str = "conv",
        finetune_stages: int = 1,
        imagenet_pretrained: bool = True,
    ):
        super().__init__()
        d = int(embed_dim)
        b = (backbone or "conv").lower()
        if b == "conv":
            h = int(hidden_dim)
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, h, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(h, 2 * h, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(2 * h, 4 * h, kernel_size=3, stride=1, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(4 * h, d),
            )
            self._backbone = None
        elif b in ("resnet18",):
            weights = models.ResNet18_Weights.DEFAULT if bool(imagenet_pretrained) and int(in_channels) == 3 else None
            resnet = models.resnet18(weights=weights)
            if int(in_channels) != 3:
                # Replace first conv to support non-RGB inputs.
                resnet.conv1 = nn.Conv2d(
                    int(in_channels),
                    resnet.conv1.out_channels,
                    kernel_size=resnet.conv1.kernel_size,
                    stride=resnet.conv1.stride,
                    padding=resnet.conv1.padding,
                    bias=False,
                )
            resnet.fc = nn.Identity()
            self._backbone = resnet
            self.proj = nn.Linear(512, d)
            self._set_resnet_trainable_stages(int(finetune_stages))
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.label_emb = nn.Embedding(int(num_classes), d)
        self.energy = nn.Linear(d, 1)

    def _set_resnet_trainable_stages(self, finetune_stages: int) -> None:
        """
        Freeze all ResNet params, then unfreeze last N stages:
          0 -> none
          1 -> layer4
          2 -> layer3+layer4
          3 -> layer2+layer3+layer4
          4 -> layer1+layer2+layer3+layer4
          5 -> all (including stem)
        """
        if self._backbone is None:
            return
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        n = int(finetune_stages)
        if n <= 0:
            return
        if n >= 5:
            for p in self._backbone.parameters():
                p.requires_grad_(True)
            return
        stages = ["layer4", "layer3", "layer2", "layer1"]
        for name in stages[: min(n, 4)]:
            m = getattr(self._backbone, name, None)
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return image feature vector f(x) without the label interaction."""
        if self._backbone is None:
            return self.encoder(x)
        return self.proj(self._backbone(x))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype != torch.long:
            y = y.to(torch.long)
        f = self.encode(x)
        e = self.label_emb(y)
        z = f * e
        out = self.energy(z)
        return out.squeeze(-1)


