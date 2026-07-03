"""
Minimal, self-contained LoRA (Low-Rank Adaptation) for the EBM.

Design goals (see the implementation plan):
  * Do NOT modify existing training/eval code. LoRA lives entirely here + in a
    dedicated runner.
  * Adapters are zero-initialized (B=0) so an injected model is IDENTICAL to the
    base model until trained.
  * `merge_lora` folds adapters back into the base weights and returns a PLAIN
    EnergyModel, so the saved checkpoint has standard keys and every existing
    evaluator (analyze_similarity_degradation.py, eval_*.py) works unchanged.

Only the backbone (features) is adapted via LoRA; the small head (proj,
label_emb, energy) is trained normally. This keeps the "we do not retrain the
feature extractor — only a low-rank correction is added" claim exact.

Verified LoRA targets for this model (from named_modules):
    proj                                 Linear (512->128)
    _backbone.layer4.0.conv1             Conv2d (256->512, 3x3, stride 2)
    _backbone.layer4.0.conv2             Conv2d (512->512, 3x3)
    _backbone.layer4.0.downsample.0      Conv2d (256->512, 1x1, stride 2)
    _backbone.layer4.1.conv1             Conv2d (512->512, 3x3)
    _backbone.layer4.1.conv2             Conv2d (512->512, 3x3)

Run the self-test:
    python -m ebm_unlearning.src.models.lora
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Convenience target groups (module names as they appear in EnergyModel.named_modules()).
LAYER4_CONV_TARGETS = [
    "_backbone.layer4.0.conv1",
    "_backbone.layer4.0.conv2",
    "_backbone.layer4.0.downsample.0",
    "_backbone.layer4.1.conv1",
    "_backbone.layer4.1.conv2",
]
LAYER3_CONV_TARGETS = [
    "_backbone.layer3.0.conv1",
    "_backbone.layer3.0.conv2",
    "_backbone.layer3.0.downsample.0",
    "_backbone.layer3.1.conv1",
    "_backbone.layer3.1.conv2",
]
PROJ_TARGET = ["proj"]


class LoRALinear(nn.Module):
    """Frozen Linear + low-rank update:  y = base(x) + scale * (x A^T) B^T."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        r = int(rank)
        self.rank = r
        self.scale = float(alpha) / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)  # B stays zero -> zero initial update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data + self.scale * (self.B @ self.A)
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data.clone()
        return merged


class LoRAConv2d(nn.Module):
    """
    Frozen Conv2d + low-rank update via a two-conv side path:
        lora_A: Conv2d(in, r, k, stride, padding, bias=False)   # same spatial reduction as base
        lora_B: Conv2d(r, out, 1x1, bias=False)                 # zero-init -> zero initial update
        y = base(x) + scale * lora_B(lora_A(x))
    Because lora_B is 1x1, the update folds exactly into a single k x k kernel.
    """

    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        r = int(rank)
        self.rank = r
        self.scale = float(alpha) / r
        self.lora_A = nn.Conv2d(base.in_channels, r, kernel_size=base.kernel_size,
                                stride=base.stride, padding=base.padding, bias=False)
        self.lora_B = nn.Conv2d(r, base.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)  # zero initial update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    @torch.no_grad()
    def merge(self) -> nn.Conv2d:
        b = self.base
        merged = nn.Conv2d(b.in_channels, b.out_channels, kernel_size=b.kernel_size,
                           stride=b.stride, padding=b.padding, dilation=b.dilation,
                           groups=b.groups, bias=b.bias is not None)
        # delta[out,in,kh,kw] = sum_r  B[out,r]  * A[r,in,kh,kw]     (B is 1x1 -> pointwise)
        b2d = self.lora_B.weight[:, :, 0, 0]                       # (out, r)
        delta = torch.einsum("or,rikl->oikl", b2d, self.lora_A.weight)  # (out, in, kh, kw)
        merged.weight.data = b.weight.data + self.scale * delta
        if b.bias is not None:
            merged.bias.data = b.bias.data.clone()
        return merged


# ── module-path helpers (handle nested + numeric indices like "layer4.0.conv1") ──
def _get_parent_and_attr(model: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
    parts = dotted.split(".")
    obj = model
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj, parts[-1]


def _set_submodule(model: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    parent, last = _get_parent_and_attr(model, dotted)
    if last.isdigit():
        parent[int(last)] = new_mod
    else:
        setattr(parent, last, new_mod)


def _get_submodule(model: nn.Module, dotted: str) -> nn.Module:
    parent, last = _get_parent_and_attr(model, dotted)
    return parent[int(last)] if last.isdigit() else getattr(parent, last)


def inject_lora(model: nn.Module, targets: List[str], rank: int, alpha: float) -> nn.Module:
    """
    Freeze the whole backbone, then wrap each target module with a LoRA adapter
    (base frozen inside, only A/B trainable). The head (proj/label_emb/energy)
    is left as-is (caller decides its requires_grad). Returns the same model.
    """
    if getattr(model, "_backbone", None) is not None:
        for p in model._backbone.parameters():
            p.requires_grad_(False)
    for name in targets:
        mod = _get_submodule(model, name)
        if isinstance(mod, nn.Conv2d):
            _set_submodule(model, name, LoRAConv2d(mod, rank, alpha))
        elif isinstance(mod, nn.Linear):
            _set_submodule(model, name, LoRALinear(mod, rank, alpha))
        else:
            raise TypeError(f"LoRA target '{name}' is {type(mod).__name__}, expected Linear/Conv2d")
    return model


@torch.no_grad()
def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every LoRA adapter into its base weights, replacing wrappers with plain
    Linear/Conv2d. After this the model has standard keys (safe to torch.save)."""
    to_merge = [(name, mod) for name, mod in model.named_modules()
                if isinstance(mod, (LoRALinear, LoRAConv2d))]
    for name, wrapper in to_merge:
        _set_submodule(model, name, wrapper.merge())
    return model


def lora_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and ("lora_A" in n or "lora_B" in n or n.endswith(".A") or n.endswith(".B")))


# ── self-test (the hard gate before using LoRA anywhere) ─────────────────────────
def _selftest() -> None:
    import copy
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ebm_unlearning.src.models.ebm import EnergyModel

    torch.manual_seed(0)
    m = EnergyModel(in_channels=3, hidden_dim=64, num_classes=26, embed_dim=128,
                    backbone="resnet18", finetune_stages=1, imagenet_pretrained=False).eval()
    x = torch.randn(4, 3, 64, 64)
    y = torch.randint(0, 26, (4,))
    base_out = m(x, y).detach().clone()

    targets = PROJ_TARGET + LAYER4_CONV_TARGETS
    lm = inject_lora(copy.deepcopy(m), targets, rank=8, alpha=16).eval()

    # (1) zero-init equality: injected == base before any training
    z = lm(x, y)
    assert torch.allclose(base_out, z, atol=1e-5), f"zero-init mismatch: {(base_out - z).abs().max()}"
    print("[lora] test 1 PASS  injected(zero-init) == base")

    # (2) simulate training: randomize adapters, then merge must match
    for n, p in lm.named_parameters():
        if p.requires_grad and (".A" in n or "lora_A" in n or ".B" in n or "lora_B" in n):
            p.data = torch.randn_like(p) * 0.05
    lm.eval()
    trained_out = lm(x, y).detach().clone()
    merged = merge_lora(copy.deepcopy(lm)).eval()
    mo = merged(x, y)
    assert torch.allclose(trained_out, mo, atol=1e-4), f"merge mismatch: {(trained_out - mo).abs().max()}"
    print("[lora] test 2 PASS  merge(trained) == trained")

    # (3) merged model is plain (no LoRA modules) and state_dict loads into a fresh EnergyModel
    assert not any(isinstance(mod, (LoRALinear, LoRAConv2d)) for _, mod in merged.named_modules())
    fresh = EnergyModel(in_channels=3, hidden_dim=64, num_classes=26, embed_dim=128,
                        backbone="resnet18", finetune_stages=1, imagenet_pretrained=False)
    missing, unexpected = fresh.load_state_dict(merged.state_dict(), strict=True)
    print("[lora] test 3 PASS  merged is plain EnergyModel, state_dict loads strict")

    print(f"[lora] trainable adapter params @ rank8 (proj+layer4): {lora_trainable_params(lm):,}")
    print("[lora] ALL TESTS PASSED")


if __name__ == "__main__":
    _selftest()
