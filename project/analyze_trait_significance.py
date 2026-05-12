#!/usr/bin/env python3
"""
Statistical significance analysis: which user traits significantly move which axes?

For each (user_type, axis) pair, runs a one-sample t-test on
projection_delta_trait_minus_neutral values and computes Cohen's d.
Applies Benjamini-Hochberg FDR correction across all (trait, axis) pairs.

Outputs two tables:
  Table 1 — all (trait, axis) pairs with p_adjusted, cohen_d, significant flag
  Table 2 — per-trait summary: n_significant_axes, mean_cohen_d, max_cohen_d, top_axis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for _p in (REPO_ROOT, PROJECT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from plots.plot_utils import load_jsonl, write_csv


# ---------------------------------------------------------------------------
# Statistics — pure Python, no scipy dependency
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-9) -> float:
    """Continued fraction for the incomplete beta function (Lentz method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-30 if abs(d) < 1e-30 else d
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c
        c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I(x; a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_p_two_sided(t: float, df: float) -> float:
    """Two-sided p-value for t-distribution."""
    x = df / (df + t * t)
    return _betai(df / 2.0, 0.5, x)


def ttest_p(values: list[float]) -> tuple[float, float]:
    """One-sample t-test against 0. Returns (t_statistic, two-sided p-value)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        return (math.inf if mean > 0 else -math.inf), 0.0
    t = mean / (std / math.sqrt(n))
    return t, _t_p_two_sided(abs(t), df=n - 1)


def cohen_d(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    if std == 0.0:
        return math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
    return mean / std


def bh_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in input order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    prev_adj = 1.0
    for rev_rank, (orig_idx, p) in enumerate(reversed(indexed)):
        rank = m - rev_rank  # 1-indexed rank among sorted p-values
        adj = min(prev_adj, p * m / rank)
        adjusted[orig_idx] = adj
        prev_adj = adj
    return adjusted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statistical significance analysis of user trait axis movement. "
            "Uses one-sample t-tests with Benjamini-Hochberg FDR correction."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Projection JSONL files (one per user trait, or mixed).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where output CSVs and JSON will be saved.",
    )
    parser.add_argument(
        "--prefix",
        default="significance",
        help="Filename prefix for generated outputs (default: significance).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min-d",
        type=float,
        default=0.2,
        help="Minimum |Cohen's d| to count as practically significant (default: 0.2).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum delta samples per (trait, axis) pair to include (default: 3).",
    )
    return parser.parse_args()


def infer_trait(row: dict[str, Any], input_path: Path) -> str | None:
    row_trait = row.get("trait")
    if isinstance(row_trait, str) and row_trait.strip():
        return row_trait.strip()
    parts = input_path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    trait_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    neutral_scores: dict[str, list[float]] = defaultdict(list)  # axis -> all neutral scores

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if input_path.stem.endswith("__neutral"):
            continue
        rows = load_jsonl(input_path)
        for row in rows:
            axis = row.get("projection_trait")
            delta = row.get("projection_delta_trait_minus_neutral")
            score_trait = row.get("projection_score_trait")
            score_neutral = row.get("projection_score_neutral")
            if axis is None or delta is None:
                continue
            trait = infer_trait(row, input_path)
            if trait is None:
                continue
            key = (trait, str(axis))
            raw_deltas[key].append(float(delta))
            if score_trait is not None:
                trait_scores[key].append(float(score_trait))
            if score_neutral is not None:
                neutral_scores[str(axis)].append(float(score_neutral))

    if not raw_deltas:
        raise ValueError("No delta values loaded from inputs.")

    # Precompute neutral mean ± std per axis across all traits
    neutral_stats: dict[str, tuple[float, float]] = {}
    for axis, vals in neutral_scores.items():
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        std = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
        neutral_stats[axis] = (m, std)

    pairs: list[dict[str, Any]] = []
    for (trait, axis), vals in raw_deltas.items():
        if len(vals) < args.min_n:
            continue
        n = len(vals)
        mean_val = sum(vals) / n
        t_stat, p_raw = ttest_p(vals)
        d = cohen_d(vals)

        # Exceedance: fraction of trait scores outside neutral mean ± 1 std
        exceedance_rate = float("nan")
        if axis in neutral_stats and trait_scores.get((trait, axis)):
            n_mean, n_std = neutral_stats[axis]
            t_scores = trait_scores[(trait, axis)]
            n_outside = sum(1 for s in t_scores if s < n_mean - n_std or s > n_mean + n_std)
            exceedance_rate = n_outside / len(t_scores)

        pairs.append(
            {
                "trait": trait,
                "axis": axis,
                "n": n,
                "mean_delta": mean_val,
                "cohen_d": d,
                "t_stat": t_stat,
                "p_raw": p_raw,
                "exceedance_rate": exceedance_rate,
            }
        )

    if not pairs:
        raise ValueError(f"No (trait, axis) pairs with n >= {args.min_n}.")

    p_adjusted_vals = bh_correction([r["p_raw"] for r in pairs])
    for row, p_adj in zip(pairs, p_adjusted_vals):
        row["p_adjusted"] = p_adj
        row["significant"] = p_adj < args.alpha and abs(row["cohen_d"]) >= args.min_d
        row["direction"] = "+" if row["mean_delta"] >= 0 else "-"

    # Table 1: all pairs, sorted by |cohen_d| descending
    table1 = sorted(pairs, key=lambda r: abs(r["cohen_d"]), reverse=True)

    # Collect significant pairs grouped by trait and by axis
    trait_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    axis_sig: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        if row["significant"]:
            trait_sig[row["trait"]].append(row)
            axis_sig[row["axis"]].append(row)

    # Table 2: per-trait summary (how many axes each trait moves significantly)
    all_traits = sorted({r["trait"] for r in pairs})
    table2: list[dict[str, Any]] = []
    for trait in all_traits:
        sig_rows = trait_sig.get(trait, [])
        n_sig = len(sig_rows)
        if sig_rows:
            abs_ds = [abs(r["cohen_d"]) for r in sig_rows]
            mean_abs_d = sum(abs_ds) / len(abs_ds)
            max_abs_d = max(abs_ds)
            top_row = max(sig_rows, key=lambda r: abs(r["cohen_d"]))
            top_axis = top_row["axis"]
            top_direction = top_row["direction"]
            exc_vals = [r["exceedance_rate"] for r in sig_rows if not math.isnan(r["exceedance_rate"])]
            mean_exceedance = sum(exc_vals) / len(exc_vals) if exc_vals else float("nan")
        else:
            mean_abs_d = 0.0
            max_abs_d = 0.0
            top_axis = ""
            top_direction = ""
            mean_exceedance = float("nan")
        table2.append(
            {
                "trait": trait,
                "composite_score": n_sig * mean_abs_d,
                "n_significant_axes": n_sig,
                "mean_abs_cohen_d": mean_abs_d,
                "max_abs_cohen_d": max_abs_d,
                "mean_exceedance_rate": mean_exceedance,
                "top_axis": top_axis,
                "top_direction": top_direction,
            }
        )
    table2 = sorted(table2, key=lambda r: r["composite_score"], reverse=True)

    # Table 3: per-axis summary (how many traits significantly move each axis)
    all_axes = sorted({r["axis"] for r in pairs})
    table3: list[dict[str, Any]] = []
    for axis in all_axes:
        sig_rows = axis_sig.get(axis, [])
        n_sig = len(sig_rows)
        if sig_rows:
            abs_ds = [abs(r["cohen_d"]) for r in sig_rows]
            mean_abs_d = sum(abs_ds) / len(abs_ds)
            max_abs_d = max(abs_ds)
            top_row = max(sig_rows, key=lambda r: abs(r["cohen_d"]))
            top_trait = top_row["trait"]
            top_direction = top_row["direction"]
            exc_vals = [r["exceedance_rate"] for r in sig_rows if not math.isnan(r["exceedance_rate"])]
            mean_exceedance = sum(exc_vals) / len(exc_vals) if exc_vals else float("nan")
        else:
            mean_abs_d = 0.0
            max_abs_d = 0.0
            top_trait = ""
            top_direction = ""
            mean_exceedance = float("nan")
        table3.append(
            {
                "axis": axis,
                "composite_score": n_sig * mean_abs_d,
                "n_significant_traits": n_sig,
                "mean_abs_cohen_d": mean_abs_d,
                "max_abs_cohen_d": max_abs_d,
                "mean_exceedance_rate": mean_exceedance,
                "top_trait": top_trait,
                "top_direction": top_direction,
            }
        )
    table3 = sorted(table3, key=lambda r: r["composite_score"], reverse=True)

    # Table 4a: distribution — how many traits move exactly N axes significantly
    axes_per_trait_counts: dict[int, int] = defaultdict(int)
    for row in table2:
        axes_per_trait_counts[row["n_significant_axes"]] += 1
    table4a = [
        {"n_significant_axes": k, "n_traits": v}
        for k, v in sorted(axes_per_trait_counts.items())
    ]

    # Table 4b: distribution — how many axes are moved by exactly N traits significantly
    traits_per_axis_counts: dict[int, int] = defaultdict(int)
    for row in table3:
        traits_per_axis_counts[row["n_significant_traits"]] += 1
    table4b = [
        {"n_significant_traits": k, "n_axes": v}
        for k, v in sorted(traits_per_axis_counts.items())
    ]

    t1_path = output_dir / f"{args.prefix}__trait_axis_pairs.csv"
    write_csv(
        table1,
        t1_path,
        fieldnames=[
            "trait", "axis", "n", "mean_delta", "cohen_d",
            "t_stat", "p_raw", "p_adjusted", "significant", "direction",
            "exceedance_rate",
        ],
    )
    print(f"Saved Table 1 (trait-axis pairs):       {t1_path}")

    t2_path = output_dir / f"{args.prefix}__trait_summary.csv"
    write_csv(
        table2,
        t2_path,
        fieldnames=[
            "trait", "composite_score", "n_significant_axes", "mean_abs_cohen_d",
            "max_abs_cohen_d", "mean_exceedance_rate", "top_axis", "top_direction",
        ],
    )
    print(f"Saved Table 2 (per-trait summary):      {t2_path}")

    t3_path = output_dir / f"{args.prefix}__axis_summary.csv"
    write_csv(
        table3,
        t3_path,
        fieldnames=[
            "axis", "composite_score", "n_significant_traits", "mean_abs_cohen_d",
            "max_abs_cohen_d", "mean_exceedance_rate", "top_trait", "top_direction",
        ],
    )
    print(f"Saved Table 3 (per-axis summary):       {t3_path}")

    t4a_path = output_dir / f"{args.prefix}__dist_traits_per_axis_count.csv"
    write_csv(table4a, t4a_path, fieldnames=["n_significant_axes", "n_traits"])
    print(f"Saved Table 4a (trait distribution):    {t4a_path}")

    t4b_path = output_dir / f"{args.prefix}__dist_axes_per_trait_count.csv"
    write_csv(table4b, t4b_path, fieldnames=["n_significant_traits", "n_axes"])
    print(f"Saved Table 4b (axis distribution):     {t4b_path}")

    json_path = output_dir / f"{args.prefix}__full.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "config": {"alpha": args.alpha, "min_d": args.min_d, "min_n": args.min_n},
                "n_pairs_tested": len(pairs),
                "n_significant_pairs": sum(1 for r in pairs if r["significant"]),
                "trait_axis_pairs": table1,
                "trait_summary": table2,
                "axis_summary": table3,
                "dist_n_axes_per_trait": table4a,
                "dist_n_traits_per_axis": table4b,
            },
            f,
            indent=2,
        )
    print(f"Saved full JSON:                        {json_path}")

    n_sig = sum(1 for r in pairs if r["significant"])
    print(
        f"\n{n_sig}/{len(pairs)} (trait, axis) pairs significant "
        f"(alpha={args.alpha}, min_d={args.min_d})"
    )

    sig_only = [r for r in table1 if r["significant"]]
    if sig_only:
        print("\nTop 10 significant pairs by |Cohen's d|:")
        for i, r in enumerate(sig_only[:10], 1):
            print(
                f"  {i:>2}. {r['trait']:<22} {r['axis']:<22} "
                f"d={r['cohen_d']:+.3f}  p_adj={r['p_adjusted']:.4f}  n={r['n']}"
            )

    print("\nTop 10 traits by composite score (n_significant_axes × mean_abs_cohen_d):")
    for i, r in enumerate(table2[:10], 1):
        top = f"{r['top_axis']}{r['top_direction']}" if r["top_axis"] else "—"
        exc = f"{r['mean_exceedance_rate']:.0%}" if not math.isnan(r["mean_exceedance_rate"]) else "n/a"
        print(
            f"  {i:>2}. {r['trait']:<22} score={r['composite_score']:>6.1f}  "
            f"n_sig={r['n_significant_axes']:>3}  mean_d={r['mean_abs_cohen_d']:.3f}  "
            f"exceedance={exc}  top={top}"
        )

    print("\nTop 10 axes by composite score (n_significant_traits × mean_abs_cohen_d):")
    for i, r in enumerate(table3[:10], 1):
        top = f"{r['top_trait']}{r['top_direction']}" if r["top_trait"] else "—"
        exc = f"{r['mean_exceedance_rate']:.0%}" if not math.isnan(r["mean_exceedance_rate"]) else "n/a"
        print(
            f"  {i:>2}. {r['axis']:<22} score={r['composite_score']:>6.1f}  "
            f"n_sig={r['n_significant_traits']:>3}  mean_d={r['mean_abs_cohen_d']:.3f}  "
            f"exceedance={exc}  top={top}"
        )

    print("\nDistribution: how many traits move exactly N axes significantly")
    for row in table4a:
        bar = "#" * min(row["n_traits"], 40)
        print(f"  {row['n_significant_axes']:>3} axes:  {row['n_traits']:>3} traits  {bar}")

    print("\nDistribution: how many axes are moved by exactly N traits significantly")
    for row in table4b:
        bar = "#" * min(row["n_axes"], 40)
        print(f"  {row['n_significant_traits']:>3} traits: {row['n_axes']:>3} axes   {bar}")


if __name__ == "__main__":
    main()
