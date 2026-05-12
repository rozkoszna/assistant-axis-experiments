#!/usr/bin/env python3
"""
Compare trait rankings by raw magnitude vs. statistical significance.

Joins the global-movers CSV (ranked by any_abs_mean_sum) with the
significance trait-summary CSV (ranked by n_significant_axes), computes
rank differences, and prints traits where the two methods disagree most.

Also computes Spearman rank correlation to quantify overall agreement.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for _p in (REPO_ROOT, PROJECT_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from plots.plot_utils import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trait rankings by raw axis movement magnitude vs. "
            "statistical significance."
        )
    )
    parser.add_argument(
        "--magnitude-csv",
        required=True,
        help="Global movers CSV (any_abs_mean_sum column). Output of analyze_trait_global_movers.py.",
    )
    parser.add_argument(
        "--significance-csv",
        required=True,
        help="Trait summary CSV (n_significant_axes column). Output of analyze_trait_significance.py.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="magnitude_vs_significance",
        help="Filename prefix (default: magnitude_vs_significance).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top discrepancies to print (default: 20).",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rank_by(rows: list[dict[str, Any]], key: str, reverse: bool = True) -> dict[str, int]:
    """Return trait -> 1-indexed rank sorted by key."""
    sorted_rows = sorted(rows, key=lambda r: float(r[key]), reverse=reverse)
    return {r["trait"]: i for i, r in enumerate(sorted_rows, 1)}


def spearman(ranks_a: list[float], ranks_b: list[float]) -> float:
    n = len(ranks_a)
    if n < 2:
        return float("nan")
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    den_a = math.sqrt(sum((a - mean_a) ** 2 for a in ranks_a))
    den_b = math.sqrt(sum((b - mean_b) ** 2 for b in ranks_b))
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mag_rows = load_csv(Path(args.magnitude_csv))
    sig_rows = load_csv(Path(args.significance_csv))

    mag_by_trait = {r["trait"]: r for r in mag_rows}
    sig_by_trait = {r["trait"]: r for r in sig_rows}

    # Only compare traits present in both
    common_traits = sorted(set(mag_by_trait) & set(sig_by_trait))
    missing_mag = sorted(set(sig_by_trait) - set(mag_by_trait))
    missing_sig = sorted(set(mag_by_trait) - set(sig_by_trait))
    if missing_mag:
        print(f"Traits in significance but not magnitude CSV: {missing_mag}")
    if missing_sig:
        print(f"Traits in magnitude but not significance CSV: {missing_sig}")

    mag_rank = rank_by(
        [mag_by_trait[t] for t in common_traits], "any_abs_mean_sum", reverse=True
    )
    sig_rank = rank_by(
        [sig_by_trait[t] for t in common_traits], "n_significant_axes", reverse=True
    )

    rows: list[dict[str, Any]] = []
    for trait in common_traits:
        mr = mag_rank[trait]
        sr = sig_rank[trait]
        rows.append(
            {
                "trait": trait,
                "magnitude_rank": mr,
                "significance_rank": sr,
                "rank_diff": sr - mr,       # positive = significance ranks lower than magnitude
                "abs_rank_diff": abs(sr - mr),
                "any_abs_mean_sum": float(mag_by_trait[trait]["any_abs_mean_sum"]),
                "n_significant_axes": int(sig_by_trait[trait]["n_significant_axes"]),
                "mean_abs_cohen_d": float(sig_by_trait[trait]["mean_abs_cohen_d"]),
            }
        )

    # Spearman correlation between the two rank lists
    ra = [mag_rank[t] for t in common_traits]
    rb = [sig_rank[t] for t in common_traits]
    rho = spearman(ra, rb)

    # Sort outputs
    by_mag = sorted(rows, key=lambda r: r["magnitude_rank"])
    by_sig = sorted(rows, key=lambda r: r["significance_rank"])
    by_diff = sorted(rows, key=lambda r: r["abs_rank_diff"], reverse=True)

    # Overrated by magnitude: high magnitude rank, low significance rank (rank_diff > 0, large)
    overrated = [r for r in by_diff if r["rank_diff"] > 0]
    # Underrated by magnitude: low magnitude rank, high significance rank (rank_diff < 0, large)
    underrated = [r for r in by_diff if r["rank_diff"] < 0]

    fieldnames = [
        "trait", "magnitude_rank", "significance_rank", "rank_diff",
        "abs_rank_diff", "any_abs_mean_sum", "n_significant_axes", "mean_abs_cohen_d",
    ]

    out_path = output_dir / f"{args.prefix}__comparison.csv"
    write_csv(by_diff, out_path, fieldnames=fieldnames)
    print(f"Saved comparison CSV: {out_path}")

    # Print summary
    print(f"\nTraits compared: {len(common_traits)}")
    print(f"Spearman rank correlation (magnitude vs significance): {rho:.4f}")

    print(f"\nTop {args.top_k} traits ranked higher by MAGNITUDE than significance")
    print("(big raw movement but not consistently significant — possibly noisy)")
    print(f"{'trait':<22} {'mag_rank':>8} {'sig_rank':>8} {'diff':>6}  "
          f"{'abs_mean_sum':>14}  {'n_sig_axes':>10}  {'mean_d':>7}")
    print("-" * 80)
    for r in overrated[: args.top_k]:
        print(
            f"  {r['trait']:<20} {r['magnitude_rank']:>8} {r['significance_rank']:>8} "
            f"{r['rank_diff']:>+6}  {r['any_abs_mean_sum']:>14.3f}  "
            f"{r['n_significant_axes']:>10}  {r['mean_abs_cohen_d']:>7.3f}"
        )

    print(f"\nTop {args.top_k} traits ranked higher by SIGNIFICANCE than magnitude")
    print("(consistent signal even if raw movement is smaller — reliable movers)")
    print(f"{'trait':<22} {'mag_rank':>8} {'sig_rank':>8} {'diff':>6}  "
          f"{'abs_mean_sum':>14}  {'n_sig_axes':>10}  {'mean_d':>7}")
    print("-" * 80)
    for r in underrated[: args.top_k]:
        print(
            f"  {r['trait']:<20} {r['magnitude_rank']:>8} {r['significance_rank']:>8} "
            f"{r['rank_diff']:>+6}  {r['any_abs_mean_sum']:>14.3f}  "
            f"{r['n_significant_axes']:>10}  {r['mean_abs_cohen_d']:>7.3f}"
        )

    print(f"\nTop 10 by magnitude ranking (for reference):")
    for r in by_mag[:10]:
        print(
            f"  mag={r['magnitude_rank']:>3}  sig={r['significance_rank']:>3}  "
            f"{r['trait']:<22}  abs_sum={r['any_abs_mean_sum']:.3f}  "
            f"n_sig={r['n_significant_axes']}"
        )

    print(f"\nTop 10 by significance ranking (for reference):")
    for r in by_sig[:10]:
        print(
            f"  sig={r['significance_rank']:>3}  mag={r['magnitude_rank']:>3}  "
            f"{r['trait']:<22}  n_sig={r['n_significant_axes']}  "
            f"mean_d={r['mean_abs_cohen_d']:.3f}"
        )


if __name__ == "__main__":
    main()
