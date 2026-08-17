"""
Turn locality JSONs into the comparison table, the frontier figure, and the
GO/NO-GO statistic.

The central claim under test is that style-invariance (DTR) and semantic
collateral damage (SCD) are driven by one entangled signal, so methods cannot
buy the first without paying the second. Operationally: across a sweep, DTR and
SCD should be POSITIVELY correlated. Spearman rho > 0.5 supports the premise;
near zero or negative refutes it.

Usage
-----
    conda run -n myn_again python analyze_locality.py \
        outputs/locality_e1_subset.json --fig outputs/frontier.png

    # group analysis for CIFAR-100 (people superclass vs the other 95)
    conda run -n myn_again python analyze_locality.py \
        outputs/locality_cifar100.json --groups people
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

CIFAR100_PEOPLE = {2: "baby", 11: "boy", 35: "girl", 46: "man", 98: "woman"}
CIFAR100_MINORS = {2, 11, 35}
CIFAR100_ADULTS = {46, 98}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsons", nargs="+", help="One or more locality JSON files from score_locality.py.")
    p.add_argument("--fig", default=None, help="Write the (SCD, DTR) frontier figure here.")
    p.add_argument("--groups", choices=["people", "auto"], default=None,
                   help="people: CIFAR-100 people superclass. auto: whichever CIFAR-100 superclass "
                        "contains the forget class — use this to check whether damage concentrates "
                        "on the target's own superclass generally, or only for people.")
    p.add_argument("--csv", default=None, help="Write the comparison table as CSV.")
    return p.parse_args()


def load_all(paths: List[Path]) -> Tuple[List[dict], dict]:
    """Flatten every report across the given JSON files into one list of rows."""
    rows: List[dict] = []
    meta: dict = {}
    for path in paths:
        blob = json.loads(path.read_text())
        meta = {k: blob[k] for k in ("class_names", "domain_names", "forget_class", "forget_domain")
                if k in blob}
        for name, rep in blob.get("reports", {}).items():
            rec = dict(rep)
            rec["name"] = name
            rec["source"] = path.name
            rows.append(rec)
    return rows, meta


def _v(x) -> float:
    return np.nan if x is None else float(x)


def parse_sweep_name(name: str):
    """
    Parse a sweep checkpoint name.

    Accepts 'k{n_pca}_lam{lambda}_s{seed}.pt' and the steps-aware form
    'k{n_pca}_lam{lambda}_st{steps}_s{seed}.pt'.
    """
    m = re.match(
        r"^k(?P<k>\d+)_lam(?P<lam>[0-9.]+)(?:_st(?P<st>\d+))?_s(?P<seed>\d+)\.pt$", name
    )
    if not m:
        return None
    return {
        "n_pca": int(m.group("k")),
        "lambda_clip": float(m.group("lam")),
        "steps": int(m.group("st")) if m.group("st") else None,
        "seed": int(m.group("seed")),
    }


KNOB_KEYS = ("lambda_clip", "steps", "n_pca")


def aggregate_sweep(rows: List[dict]):
    """
    Group sweep points by their hyperparameters and aggregate over seeds.

    The swept "knob" is auto-detected as whichever of lambda_clip / steps / n_pca
    actually varies, so the same analysis serves a lambda sweep and a duration
    sweep. Returns (aggregated rows, knob name), or None if these are not sweep
    checkpoints.
    """
    parsed = [(parse_sweep_name(r["name"]), r) for r in rows]
    if any(p is None for p, _ in parsed):
        return None

    knob = None
    for key in KNOB_KEYS:
        vals = {p[key] for p, _ in parsed if p[key] is not None}
        if len(vals) > 1:
            knob = key
            break
    if knob is None:
        knob = "lambda_clip"

    groups: Dict[Tuple, List[dict]] = {}
    for p, r in parsed:
        groups.setdefault(tuple(p[k] for k in KNOB_KEYS), []).append((p, r))

    out = []
    for key, items in groups.items():
        p0 = items[0][0]
        rs = [r for _, r in items]
        rec = {k: p0[k] for k in KNOB_KEYS}
        rec["n_seeds"] = len(rs)
        rec["knob_value"] = p0[knob]
        for k in ("dtr", "scd", "scd_weighted", "retain_utility", "target_forgetting"):
            vals = np.array([_v(r.get(k)) for r in rs], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            rec[k] = float(vals.mean()) if vals.size else np.nan
            rec[k + "_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out.append(rec)
    out.sort(key=lambda r: (r["knob_value"] if r["knob_value"] is not None else 0))
    return out, knob


def print_sweep_table(agg: List[dict], knob: str) -> None:
    print("=" * 100)
    print(f"  Sweep over {knob} (mean +/- std over seeds)")
    print("=" * 100)
    print(f"  {knob:>10} {'n':>3}  {'F_target':>15} {'DTR':>15} {'SCD':>15} {'Utility':>9}")
    print("  " + "-" * 92)
    for r in agg:
        def pm(key, pct=False):
            m, s = r[key], r[key + "_std"]
            if not np.isfinite(m):
                return "          n/a"
            return f"{m:6.1%}+/-{s:4.1%}" if pct else f"{m:6.3f}+/-{s:5.3f}"
        print(f"  {r['knob_value']:>10g} {r['n_seeds']:>3}  {pm('target_forgetting', True)} "
              f"{pm('dtr')} {pm('scd')} {r['retain_utility']:8.1%}")
    print("=" * 100)
    n_nan = sum(1 for r in agg if not np.isfinite(r["dtr"]))
    if n_nan:
        print(f"  ! DTR withheld at {n_nan}/{len(agg)} setting(s): target cell under-erased, so the")
        print("    DTR ratio would be unstable. Those rows cannot be used for the trade-off claim.")


def go_no_go_sweep(agg: List[dict], rows: List[dict], knob: str = "lambda_clip") -> None:
    """Premise test on a clean single-parameter sweep."""
    print()
    print("=" * 96)
    print(f"  GO / NO-GO — single-parameter sweep over {knob}")
    print("=" * 96)
    lam = np.array([r["knob_value"] for r in agg], dtype=np.float64)
    dtr = np.array([r["dtr"] for r in agg])
    scd = np.array([r["scd"] for r in agg])
    ok = np.isfinite(dtr) & np.isfinite(scd)
    try:
        from scipy.stats import spearmanr
        rho_m, p_m = spearmanr(scd[ok], dtr[ok])
        rl_d, _ = spearmanr(lam[ok], dtr[ok])
        rl_s, _ = spearmanr(lam[ok], scd[ok])
    except Exception:
        rho_m = p_m = rl_d = rl_s = float("nan")
    print(f"  {knob} -> DTR   Spearman rho = {rl_d:+.3f}   (does the knob buy style-invariance?)")
    print(f"  {knob} -> SCD   Spearman rho = {rl_s:+.3f}   (does the same knob cost neighbours?)")
    print(f"  SCD <-> DTR     Spearman rho = {rho_m:+.3f}   p = {p_m:.4f}   [n={int(ok.sum())} lambda values]")

    pts = [(_v(r.get("scd")), _v(r.get("dtr"))) for r in rows]
    pts = [(s, d) for s, d in pts if np.isfinite(s) and np.isfinite(d)]
    if len(pts) >= 4:
        try:
            from scipy.stats import spearmanr
            rho_a, p_a = spearmanr(np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
            print(f"  SCD <-> DTR     Spearman rho = {rho_a:+.3f}   p = {p_a:.4f}   "
                  f"[n={len(pts)} individual runs]")
        except Exception:
            pass
    # A positive correlation between the two aggregated means is NOT sufficient. If DTR
    # barely moves, that correlation is an artifact of a handful of noisy means and the
    # knob is not actually buying style-invariance. Require, in order:
    #   (a) DTR genuinely varies -- its spread across the knob must exceed the seed noise
    #   (b) the knob raises DTR
    #   (c) the knob raises SCD
    #   (d) the per-run correlation agrees in sign with the aggregated one
    dtr_range = float(np.nanmax(dtr[ok]) - np.nanmin(dtr[ok])) if ok.any() else np.nan
    seed_noise = float(np.nanmean([r["dtr_std"] for r in agg]))
    moves = np.isfinite(dtr_range) and np.isfinite(seed_noise) and dtr_range > 2.0 * seed_noise
    print()
    print(f"  DTR spread across knob = {dtr_range:.3f}   mean seed std = {seed_noise:.3f}"
          f"   -> DTR {'moves' if moves else 'is FLAT within noise'}")
    print()
    if not moves:
        print("  NO-GO (knob) — DTR does not vary beyond seed noise, so this parameter does")
        print("       not buy style-invariance at these settings. Any SCD<->DTR correlation")
        print("       across the aggregated means is an artifact of a few noisy points.")
        print("       Sweep the parameter that actually drives target forgetting (duration),")
        print("       and check that F[c*,d*] converges before comparing DTR.")
    elif rl_d > 0.5 and rl_s > 0.5 and rho_m > 0.5:
        print("  GO — within a single method, turning the knob up buys style-invariance and")
        print("       pays for it in collateral damage. The axes are entangled.")
    elif rho_m > 0.2:
        print("  WEAK — positive but under threshold. Widen the range or add seeds.")
    else:
        print("  NO-GO — the axes do not co-move within the sweep. The premise as stated")
        print("          does not hold; fall back to D6 (relearning robustness).")
    print("=" * 96)


def plot_sweep_frontier(agg: List[dict], rows: List[dict], out: Path, knob: str = "lambda_clip") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # left: the frontier, mean +/- std, ordered by lambda
    ax = axes[0]
    scd = np.array([r["scd"] for r in agg]);  scd_s = np.array([r["scd_std"] for r in agg])
    dtr = np.array([r["dtr"] for r in agg]);  dtr_s = np.array([r["dtr_std"] for r in agg])
    lam = np.array([r["knob_value"] for r in agg], dtype=np.float64)
    o = np.argsort(lam)
    ax.errorbar(scd[o], dtr[o], xerr=scd_s[o], yerr=dtr_s[o], fmt="-o", color="#2b6cb0",
                ecolor="#a0aec0", capsize=3, lw=1.6, ms=7, zorder=3)
    for i in o:
        ax.annotate(f"{lam[i]:g}", (scd[i], dtr[i]), fontsize=8.5,
                    xytext=(7, -3), textcoords="offset points", color="#2d3748")
    ax.axhline(1.0, ls="--", lw=1, c="#38a169", zorder=1)
    ax.set_xlabel("SCD — semantic collateral damage  (want 0)")
    ax.set_ylabel("DTR — style-invariance  (want 1)")
    ax.set_title("Erasure strength trades one axis against the other")
    ax.grid(alpha=0.25, zorder=0)

    # right: both axes against the knob
    ax2 = axes[1]
    ax2.errorbar(lam[o], dtr[o], yerr=dtr_s[o], fmt="-o", color="#38a169", capsize=3, label="DTR (want high)")
    ax2.errorbar(lam[o], scd[o], yerr=scd[o] * 0 + np.array([r["scd_std"] for r in agg])[o],
                 fmt="-s", color="#e53e3e", capsize=3, label="SCD (want low)")
    ax2.set_xlabel(f"{knob}  (erasure strength)")
    ax2.set_ylabel("metric value")
    ax2.set_title("Both axes move together with the same knob")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"\n  figure -> {out}")


def print_table(rows: List[dict]) -> None:
    print("=" * 96)
    print("  TWO-SIDED LOCALITY — method comparison")
    print("=" * 96)
    print(f"  {'checkpoint':46} {'DTR':>7} {'SCD':>8} {'SCD_w':>8} {'Sel':>7} {'Util':>7}")
    print("  " + "-" * 90)
    for r in sorted(rows, key=lambda r: (-_v(r.get("selectivity")) if np.isfinite(_v(r.get("selectivity"))) else 1e9)):
        def f(k, pct=False):
            v = _v(r.get(k))
            if not np.isfinite(v):
                return "    n/a"
            return f"{v:6.1%}" if pct else f"{v:7.3f}"
        print(f"  {r['name'][:46]:46} {f('dtr')} {f('scd')} {f('scd_weighted')} "
              f"{f('selectivity')} {f('retain_utility', True)}")
    print("=" * 96)
    print("  DTR -> 1 is good (erasure transfers across style).")
    print("  SCD -> 0 is good (erasure spares semantic neighbours).")
    print("  Sel = DTR - SCD. Util = retain accuracy post/pre.")


def go_no_go(rows: List[dict]) -> None:
    """The premise test: are the two axes entangled?"""
    pairs = [(_v(r.get("scd")), _v(r.get("dtr")), r["name"]) for r in rows]
    pairs = [(s, d, n) for s, d, n in pairs if np.isfinite(s) and np.isfinite(d)]
    print()
    print("=" * 96)
    print("  GO / NO-GO — is style-invariance entangled with collateral damage?")
    print("=" * 96)
    if len(pairs) < 4:
        print(f"  Only {len(pairs)} scorable point(s); need >= 4 for a meaningful correlation.")
        print("  -> run the lambda_clip sweep (E2) before deciding.")
        return
    scd = np.array([p[0] for p in pairs])
    dtr = np.array([p[1] for p in pairs])
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(scd, dtr)
    except Exception:
        # rank correlation without scipy
        rs, rd = np.argsort(np.argsort(scd)), np.argsort(np.argsort(dtr))
        rho = float(np.corrcoef(rs, rd)[0, 1])
        pval = float("nan")
    print(f"  n = {len(pairs)} scorable checkpoints")
    print(f"  Spearman rho(SCD, DTR) = {rho:+.3f}" + (f"   p = {pval:.4f}" if np.isfinite(pval) else ""))
    print()
    if rho > 0.5:
        print("  GO — the axes are entangled as hypothesised. Every gain in style-invariance")
        print("       is bought with collateral damage. Proceed to E2/E5.")
    elif rho > 0.2:
        print("  WEAK — positive but under the 0.5 threshold. Run the E2 sweep before")
        print("         committing; heterogeneous methods are a noisier test than one sweep.")
    else:
        print("  NO-GO — the axes do not co-move across these checkpoints. Either the")
        print("          premise is wrong or these methods differ on too many axes at once.")
        print("          Run E2 (a clean single-parameter sweep) before falling back to D6.")
    print("=" * 96)


def plot_frontier(rows: List[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [(_v(r.get("scd")), _v(r.get("dtr")), r["name"]) for r in rows]
    pts = [(s, d, n) for s, d, n in pts if np.isfinite(s) and np.isfinite(d)]
    if not pts:
        print("  ! nothing plottable (no checkpoint has both DTR and SCD)")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scd = np.array([p[0] for p in pts])
    dtr = np.array([p[1] for p in pts])
    ax.scatter(scd, dtr, s=90, c="#2b6cb0", zorder=3, edgecolors="white", linewidths=1.5)
    for s, d, n in pts:
        label = n.replace("ebm_", "").replace("_subset", "").replace(".pt", "")
        ax.annotate(label, (s, d), fontsize=7.5, xytext=(6, 4),
                    textcoords="offset points", color="#333333")

    ax.axhline(1.0, ls="--", lw=1, c="#38a169", zorder=1)
    ax.axvline(0.0, ls="--", lw=1, c="#e53e3e", zorder=1)
    ax.text(ax.get_xlim()[1], 1.0, " ideal DTR", va="bottom", ha="right", fontsize=8, color="#38a169")

    ax.set_xlabel("SCD — semantic collateral damage  (want 0, left is better)")
    ax.set_ylabel("DTR — style-invariance  (want 1, up is better)")
    ax.set_title("Two-sided locality: erasure must transfer across style, not across concept")
    ax.grid(alpha=0.25, zorder=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"\n  figure -> {out}")


def group_analysis_superclass(rows: List[dict], meta: dict) -> None:
    """
    Generalised control: compare damage inside the forget class's OWN CIFAR-100
    superclass against everything else.

    This is the check a reviewer will ask for. If erasing `wolf` concentrates on
    large_carnivores just as strongly as erasing `boy` concentrates on people,
    then the effect is a generic property of semantic neighbourhoods rather than
    anything specific to person concepts -- and the child-safety claim has to be
    stated as "this generic mechanism happens to fall on the representations
    child-protection tooling needs", not "children are special".
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ebm_unlearning.src.data.split import CIFAR100_SUPERCLASS_TO_FINE

    class_names = meta.get("class_names") or []
    print()
    print("=" * 96)
    print("  CONTROL — does damage concentrate on the forget class's own superclass?")
    print("=" * 96)
    print(f"  {'checkpoint':34} {'target':>10} {'own superclass':>16} {'others':>10} {'ratio':>8}")
    print("  " + "-" * 90)
    for r in rows:
        F = r.get("forgetting")
        if F is None:
            continue
        arr = np.array([[np.nan if v is None else v for v in row] for row in F], dtype=np.float64)
        per_class = np.array([np.nanmean(row) if np.isfinite(row).any() else np.nan for row in arr])
        fc = r.get("forget_class", -1)
        sc_members = None
        for _sc, fine in CIFAR100_SUPERCLASS_TO_FINE.items():
            if fc in fine:
                sc_members = set(fine)
                break
        if sc_members is None:
            continue
        sib = [c for c in sc_members if c != fc and c < len(per_class)]
        oth = [c for c in range(len(per_class)) if c != fc and c not in sc_members]
        sv = float(np.nanmean(per_class[sib])) if sib else np.nan
        ov = float(np.nanmean(per_class[oth])) if oth else np.nan
        ratio = sv / ov if np.isfinite(sv) and np.isfinite(ov) and abs(ov) > 1e-6 else np.nan
        tgt = class_names[fc] if 0 <= fc < len(class_names) else str(fc)
        print(f"  {r['name'][:34]:34} {tgt:>10} {sv:15.1%} {ov:9.1%} {ratio:7.2f}x")
    print("=" * 96)


def group_analysis_people(rows: List[dict], meta: dict) -> None:
    """
    CIFAR-100: is damage concentrated on the person manifold?

    Compares mean forgetting over the people superclass (minus the forget class)
    against the other 95 classes. This is the E4 result.
    """
    class_names = meta.get("class_names") or []
    print()
    print("=" * 96)
    print("  E4 — is collateral damage concentrated on the person manifold?")
    print("=" * 96)
    print(f"  {'checkpoint':40} {'people (excl. target)':>22} {'other classes':>16} {'ratio':>8}")
    print("  " + "-" * 90)
    for r in rows:
        F = r.get("forgetting")
        if F is None:
            continue
        arr = np.array([[np.nan if v is None else v for v in row] for row in F], dtype=np.float64)
        per_class = np.array([np.nanmean(row) if np.isfinite(row).any() else np.nan for row in arr])
        fc = r.get("forget_class", -1)
        people = [c for c in CIFAR100_PEOPLE if c != fc and c < len(per_class)]
        others = [c for c in range(len(per_class)) if c != fc and c not in CIFAR100_PEOPLE]
        pv = float(np.nanmean(per_class[people])) if people else np.nan
        ov = float(np.nanmean(per_class[others])) if others else np.nan
        ratio = pv / ov if np.isfinite(pv) and np.isfinite(ov) and abs(ov) > 1e-6 else np.nan
        tgt = class_names[fc] if 0 <= fc < len(class_names) else str(fc)
        print(f"  {r['name'][:40]:40} {pv:21.1%} {ov:15.1%} {ratio:7.2f}x   (target: {tgt})")
        # per-person-class detail
        detail = "      "
        for c in sorted(CIFAR100_PEOPLE):
            if c == fc or c >= len(per_class):
                continue
            if np.isfinite(per_class[c]):
                detail += f"{CIFAR100_PEOPLE[c]}={per_class[c]:+.1%}  "
        if detail.strip():
            print(detail)
    print("=" * 96)
    print("  ratio > 1 means erasure leaks preferentially into other person classes,")
    print("  i.e. the damage tracks semantic adjacency rather than being uniform.")


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.jsons]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing json(s): {', '.join(str(m) for m in missing)}")

    rows, meta = load_all(paths)
    if not rows:
        raise SystemExit("no reports found in the given JSON file(s)")

    parsed = aggregate_sweep(rows)
    if parsed is not None:
        agg, knob = parsed
        print_sweep_table(agg, knob)
        go_no_go_sweep(agg, rows, knob)
        if args.fig:
            plot_sweep_frontier(agg, rows, Path(args.fig), knob)
    else:
        print_table(rows)
        go_no_go(rows)
        if args.fig:
            plot_frontier(rows, Path(args.fig))

    if args.groups == "people":
        group_analysis_people(rows, meta)
    elif args.groups == "auto":
        group_analysis_superclass(rows, meta)

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["name", "dtr", "scd", "scd_weighted", "selectivity", "retain_utility", "target_forgetting"]
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
        out.write_text("\n".join(lines))
        print(f"  csv -> {out}")


if __name__ == "__main__":
    main()
