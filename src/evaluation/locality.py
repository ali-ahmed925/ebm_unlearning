"""
Two-sided locality metrics for concept erasure.

Motivation (child-safety framing): erasing a harmful concept carries two
*opposing* requirements that are usually conflated.

  1. Style-invariance   — erasure applied in one rendering domain MUST transfer
                          to the same concept in every other domain. A harmful
                          depiction is harmful whether photographic, sketched,
                          clipart, or painted.
  2. Semantic locality  — erasure MUST NOT propagate to semantically adjacent
                          benign concepts, because destroying benign
                          representation breaks the downstream tooling that
                          depends on it.

We measure (1) with DTR and (2) with SCD, over a (class x domain) accuracy grid
computed before and after unlearning.

Notation. Forget target is the cell (c*, d*). For every cell we define the
relative forgetting

    F[c][d] = (A_pre[c][d] - A_post[c][d]) / A_pre[c][d]

so F = 1 means the cell was fully forgotten and F = 0 means untouched. F may be
negative when a cell improves; we do not clip, so the reported numbers stay
faithful.

    DTR = mean_{d != d*} F[c*][d]  /  F[c*][d*]        -> want 1 (style-invariant)
    SCD = mean_{c != c*} mean_d F[c][d]                 -> want 0 (semantically local)

DTR is undefined when the target cell was not actually forgotten
(F[c*][d*] <= 0); we return NaN rather than a misleading ratio.

Both metrics deliberately exclude the forget class from SCD: transfer of
erasure to c* in other domains is the *desired* behaviour and must not be
charged as collateral damage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_EPS = 1e-8


# ── Fast exact all-class energies ─────────────────────────────────────────────

@torch.no_grad()
def all_class_energies(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Energies for every (sample, label) pair, using the bilinear factorization.

    EnergyModel computes E(x, y) = W . (f(x) * e_y) + b, so

        E(x, .) = f(x) @ (W * e).T + b

    i.e. all C label energies follow from a SINGLE backbone pass. The naive
    implementation in classification.predict_argmin_energy replicates x once per
    label and is therefore C times slower. This routine is exact, not an
    approximation -- `verify_against_reference` below asserts agreement.

    Returns (energies (N, C) float32, y_true (N,) int64).
    """
    model.eval()
    W = model.energy.weight[0]                    # (d,)
    b = model.energy.bias                         # (1,)
    G = model.label_emb.weight * W.unsqueeze(0)   # (C, d)

    all_e: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []
    for batch in loader:
        xb, yb = batch[0], batch[1]
        f = model.encode(xb.to(device))           # (B, d)
        all_e.append((f @ G.T + b).float().cpu())
        all_y.append(yb.cpu())
    return torch.cat(all_e).numpy(), torch.cat(all_y).to(torch.long).numpy()


@torch.no_grad()
def verify_against_reference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int = 2,
    tol: float = 1e-4,
) -> float:
    """
    Sanity check: the factorized energies must match a direct E(x, y) call.

    Returns the max absolute difference. Raises if it exceeds `tol`.
    """
    model.eval()
    W = model.energy.weight[0]
    b = model.energy.bias
    G = model.label_emb.weight * W.unsqueeze(0)

    worst = 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        xb = batch[0].to(device)
        fast = model.encode(xb) @ G.T + b
        naive = torch.stack(
            [model(xb, torch.full((xb.shape[0],), c, dtype=torch.long, device=device))
             for c in range(num_classes)],
            dim=1,
        )
        worst = max(worst, float((fast - naive).abs().max()))
    if worst > tol:
        raise AssertionError(
            f"Factorized energies disagree with direct E(x,y) by {worst:.3e} (> {tol:.1e}). "
            "The model's energy head is no longer a plain Linear over f(x)*e_y."
        )
    return worst


# ── Cell grid ─────────────────────────────────────────────────────────────────

def per_cell_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    domain_idx: Optional[np.ndarray],
    num_classes: int,
    num_domains: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accuracy and sample count for every (class, domain) cell.

    `domain_idx=None` collapses to a single pseudo-domain, which is how
    single-domain datasets (CIFAR-100) reuse this module.

    Returns (accuracy (C, D) with NaN for empty cells, counts (C, D) int).
    """
    if domain_idx is None:
        domain_idx = np.zeros_like(y_true)
    acc = np.full((num_classes, num_domains), np.nan, dtype=np.float64)
    cnt = np.zeros((num_classes, num_domains), dtype=np.int64)
    correct = (y_true == y_pred)
    for c in range(num_classes):
        cm = (y_true == c)
        for d in range(num_domains):
            m = cm & (domain_idx == d)
            n = int(m.sum())
            cnt[c, d] = n
            if n:
                acc[c, d] = float(correct[m].mean())
    return acc, cnt


def relative_forgetting(acc_pre: np.ndarray, acc_post: np.ndarray) -> np.ndarray:
    """F = (A_pre - A_post) / A_pre, NaN where A_pre is 0 or missing."""
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (acc_pre - acc_post) / acc_pre
    f[~np.isfinite(f)] = np.nan
    f[np.isclose(acc_pre, 0.0)] = np.nan
    return f


# ── Report ────────────────────────────────────────────────────────────────────

@dataclass
class LocalityReport:
    """Two-sided locality summary for one unlearned checkpoint."""

    dtr: float                                  # style-invariance, want 1
    scd: float                                  # semantic collateral damage, want 0
    scd_weighted: Optional[float]               # similarity-weighted SCD
    selectivity: float                          # dtr - scd, single-number summary
    target_forgetting: float                    # F[c*][d*]
    transfer_forgetting: Dict[str, float]       # F[c*][d] for d != d*
    retain_utility: float                       # mean A_post / mean A_pre over c != c*
    forget_class: int = -1
    forget_domain: Optional[int] = None
    class_names: Sequence[str] = field(default_factory=list)
    domain_names: Sequence[str] = field(default_factory=list)
    acc_pre: Optional[np.ndarray] = None
    acc_post: Optional[np.ndarray] = None
    forgetting: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable form (arrays become nested lists, NaN -> None)."""
        def _arr(a):
            if a is None:
                return None
            return [[None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
                    for row in a.tolist()]

        def _f(v):
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        return {
            "dtr": _f(self.dtr),
            "scd": _f(self.scd),
            "scd_weighted": _f(self.scd_weighted),
            "selectivity": _f(self.selectivity),
            "target_forgetting": _f(self.target_forgetting),
            "transfer_forgetting": {k: _f(v) for k, v in self.transfer_forgetting.items()},
            "retain_utility": _f(self.retain_utility),
            "forget_class": self.forget_class,
            "forget_domain": self.forget_domain,
            "class_names": list(self.class_names),
            "domain_names": list(self.domain_names),
            "acc_pre": _arr(self.acc_pre),
            "acc_post": _arr(self.acc_post),
            "forgetting": _arr(self.forgetting),
            "counts": None if self.counts is None else self.counts.tolist(),
            "notes": list(self.notes),
        }


def compute_locality(
    acc_pre: np.ndarray,
    acc_post: np.ndarray,
    counts: np.ndarray,
    forget_class: int,
    forget_domain: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    domain_names: Optional[Sequence[str]] = None,
    similarities: Optional[np.ndarray] = None,
    min_cell_count: int = 10,
    min_target_forgetting: float = 0.9,
) -> LocalityReport:
    """
    Build the two-sided locality report from pre/post (class, domain) accuracy grids.

    similarities: optional (C,) semantic similarity of each class to the forget
    class, used for the similarity-weighted SCD. Entry for the forget class is
    ignored.
    """
    num_classes, num_domains = acc_pre.shape
    class_names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    domain_names = list(domain_names) if domain_names is not None else [str(i) for i in range(num_domains)]
    notes: List[str] = []

    F = relative_forgetting(acc_pre, acc_post)

    thin = int(((counts > 0) & (counts < min_cell_count)).sum())
    if thin:
        notes.append(f"{thin} cell(s) have fewer than {min_cell_count} samples; treat those rows as noisy.")

    # ── Style-invariance (DTR) ────────────────────────────────────────────────
    transfer: Dict[str, float] = {}
    if forget_domain is None or num_domains == 1:
        target_f = float(F[forget_class, 0]) if num_domains == 1 else np.nan
        dtr = np.nan
        notes.append("Single-domain dataset: DTR is undefined (no style axis).")
    else:
        target_f = float(F[forget_class, forget_domain])
        others = [d for d in range(num_domains) if d != forget_domain]
        for d in others:
            transfer[domain_names[d]] = float(F[forget_class, d])
        vals = np.array([F[forget_class, d] for d in others], dtype=np.float64)
        mean_transfer = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        if not np.isfinite(target_f) or target_f <= 0:
            dtr = np.nan
            notes.append(
                f"Target cell was not forgotten (F[c*][d*]={target_f:.4f}); DTR undefined."
            )
        elif target_f < min_target_forgetting:
            # DTR is transfer NORMALISED by target forgetting. When the target cell is
            # only partly erased the denominator is itself noisy, and DTR becomes a ratio
            # of two unstable quantities -- it then anti-correlates with F[c*][d*] for
            # purely arithmetic reasons. Refuse it rather than report an unstable number.
            dtr = np.nan
            notes.append(
                f"Target cell only {target_f:.1%} forgotten (< {min_target_forgetting:.0%} required); "
                "DTR withheld as unstable. Increase erasure duration before comparing DTR."
            )
        else:
            dtr = mean_transfer / target_f
        # Absolute transfer is reported unconditionally: unlike DTR it needs no
        # denominator, so it stays interpretable even when erasure is incomplete.
        transfer["_mean_absolute"] = mean_transfer

    # ── Semantic locality (SCD) ───────────────────────────────────────────────
    retain_classes = [c for c in range(num_classes) if c != forget_class]
    per_class = np.array(
        [np.nanmean(F[c]) if np.isfinite(F[c]).any() else np.nan for c in retain_classes],
        dtype=np.float64,
    )
    scd = float(np.nanmean(per_class)) if np.isfinite(per_class).any() else np.nan

    scd_weighted: Optional[float] = None
    if similarities is not None:
        s = np.asarray(similarities, dtype=np.float64)[retain_classes]
        ok = np.isfinite(per_class) & np.isfinite(s)
        if ok.any() and s[ok].sum() > _EPS:
            scd_weighted = float((per_class[ok] * s[ok]).sum() / s[ok].sum())

    # ── Utility ───────────────────────────────────────────────────────────────
    pre_r = acc_pre[retain_classes]
    post_r = acc_post[retain_classes]
    ok = np.isfinite(pre_r) & np.isfinite(post_r)
    retain_utility = float(post_r[ok].mean() / pre_r[ok].mean()) if ok.any() and pre_r[ok].mean() > _EPS else np.nan

    selectivity = float(dtr - scd) if np.isfinite(dtr) and np.isfinite(scd) else np.nan

    return LocalityReport(
        dtr=dtr,
        scd=scd,
        scd_weighted=scd_weighted,
        selectivity=selectivity,
        target_forgetting=target_f,
        transfer_forgetting=transfer,
        retain_utility=retain_utility,
        forget_class=forget_class,
        forget_domain=forget_domain,
        class_names=class_names,
        domain_names=domain_names,
        acc_pre=acc_pre,
        acc_post=acc_post,
        forgetting=F,
        counts=counts,
        notes=notes,
    )


# ── Presentation ──────────────────────────────────────────────────────────────

def format_report(report: LocalityReport, top_k: int = 8) -> str:
    """Human-readable summary: headline metrics, transfer row, worst-damaged classes."""
    L: List[str] = []
    fc = report.class_names[report.forget_class] if report.class_names else str(report.forget_class)
    fd = (report.domain_names[report.forget_domain]
          if report.forget_domain is not None and report.domain_names else "-")

    def fmt(v, pct=True):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "  n/a"
        return f"{v:6.1%}" if pct else f"{v:6.3f}"

    L.append("=" * 66)
    L.append(f"  TWO-SIDED LOCALITY — forget: {fc}" + (f" ({fd})" if fd != "-" else ""))
    L.append("=" * 66)
    L.append(f"  {'DTR  — style-invariance (want 1.00)':45} {fmt(report.dtr, False)}")
    L.append(f"  {'SCD  — semantic collateral damage (want 0)':45} {fmt(report.scd, False)}")
    if report.scd_weighted is not None:
        L.append(f"  {'SCD_w — similarity-weighted':45} {fmt(report.scd_weighted, False)}")
    L.append(f"  {'Selectivity = DTR - SCD (higher better)':45} {fmt(report.selectivity, False)}")
    L.append("-" * 66)
    L.append(f"  {'Forgetting at target cell':45} {fmt(report.target_forgetting)}")
    for d, v in report.transfer_forgetting.items():
        label = "  mean absolute transfer" if d == "_mean_absolute" else "  transfer -> " + d
        L.append(f"  {label:45} {fmt(v)}")
    L.append(f"  {'Retain utility (post/pre)':45} {fmt(report.retain_utility)}")

    if report.forgetting is not None:
        L.append("-" * 66)
        L.append(f"  Most-damaged retain classes (mean forgetting across domains):")
        rows = []
        for c in range(len(report.class_names)):
            if c == report.forget_class:
                continue
            row = report.forgetting[c]
            if not np.isfinite(row).any():
                continue  # class absent from this eval split
            v = float(np.nanmean(row))
            rows.append((v, report.class_names[c]))
        rows.sort(reverse=True)
        for v, name in rows[:top_k]:
            bar = "#" * min(30, max(0, int(round(v * 30))))
            L.append(f"    {name:22} {v:7.1%}  {bar}")

    for n in report.notes:
        L.append(f"  ! {n}")
    L.append("=" * 66)
    return "\n".join(L)


def format_cell_table(report: LocalityReport, max_classes: int = 40) -> str:
    """Full per-class per-domain pre/post accuracy grid."""
    L: List[str] = []
    C = len(report.class_names)
    L.append("")
    L.append(f"  {'Class':22} {'Domain':10} {'Pre':>8} {'Post':>8} {'Delta':>9}")
    L.append("  " + "-" * 60)
    for c in range(min(C, max_classes)):
        mark = "  <- FORGET CLASS" if c == report.forget_class else ""
        for d, dn in enumerate(report.domain_names):
            pre, post = report.acc_pre[c, d], report.acc_post[c, d]
            if not np.isfinite(pre):
                continue
            tag = mark if (c == report.forget_class and d == (report.forget_domain or 0)) else ""
            L.append(f"  {report.class_names[c]:22} {dn:10} {pre:7.1%} {post:7.1%} "
                     f"{post - pre:+8.1%}{tag}")
        L.append("  " + "-" * 60)
    return "\n".join(L)
