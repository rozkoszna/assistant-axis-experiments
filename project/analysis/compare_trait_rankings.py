#!/usr/bin/env python3
"""
Compare how two CSVs rank traits and find where they disagree.

Takes any two CSVs that each have a 'trait' column and a numeric score column,
ranks traits by those scores, and reports where the rankings diverge most.
Computes Spearman rank correlation to quantify overall agreement.

Common use cases:
- Two runs of the same experiment (do trait rankings replicate?)
- Magnitude vs. significance (any_abs_mean_sum vs. n_significant_axes)
- Two different intent types (identity vs. natural questions)

Example:
  python project/analysis/compare_trait_rankings.py \\
    --csv-a  outputs/analysis/trait_movers/trait_global_movers__any_abs_mean_sum.csv \\
    --col-a  any_abs_mean_sum \\
    --label-a magnitude \\
    --csv-b  outputs/analysis/significance/strict_all_axes__trait_summary.csv \\
    --col-b  n_significant_axes \\
    --label-b significance \\
    --output-dir outputs/analysis/trait_rankings/
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
        description="Compare how two CSVs rank traits and find where they disagree."
    )
    parser.add_argument("--csv-a", required=True, help="First CSV (must have a 'trait' column).")
    parser.add_argument("--col-a", required=True, help="Numeric column in csv-a to rank by.")
    parser.add_argument("--label-a", default="a", help="Label for the first ranking in output (default: a).")
    parser.add_argument("--csv-b", required=True, help="Second CSV (must have a 'trait' column).")
    parser.add_argument("--col-b", required=True, help="Numeric column in csv-b to rank by.")
    parser.add_argument("--label-b", default="b", help="Label for the second ranking in output (default: b).")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison outputs.")
    parser.add_argument("--prefix", default="trait_rankings", help="Filename prefix (default: trait_rankings).")
    parser.add_argument("--top-k", type=int, default=20, help="How many top discrepancies to print (default: 20).")
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def rank_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    """Return trait -> 1-indexed rank, highest score = rank 1."""
    sorted_rows = sorted(rows, key=lambda r: float(r[key]), reverse=True)
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

    rows_a = load_csv(Path(args.csv_a))
    rows_b = load_csv(Path(args.csv_b))

    by_trait_a = {r["trait"]: r for r in rows_a}
    by_trait_b = {r["trait"]: r for r in rows_b}

    common_traits = sorted(set(by_trait_a) & set(by_trait_b))
    only_in_a = sorted(set(by_trait_a) - set(by_trait_b))
    only_in_b = sorted(set(by_trait_b) - set(by_trait_a))
    if only_in_a:
        print(f"Traits only in {args.label_a}: {only_in_a}")
    if only_in_b:
        print(f"Traits only in {args.label_b}: {only_in_b}")

    rank_a = rank_by([by_trait_a[t] for t in common_traits], args.col_a)
    rank_b = rank_by([by_trait_b[t] for t in common_traits], args.col_b)

    rows: list[dict[str, Any]] = []
    for trait in common_traits:
        ra = rank_a[trait]
        rb = rank_b[trait]
        rows.append(
            {
                "trait": trait,
                f"rank_{args.label_a}": ra,
                f"rank_{args.label_b}": rb,
                "rank_diff": rb - ra,        # positive = b ranks this trait lower than a
                "abs_rank_diff": abs(rb - ra),
                f"{args.col_a}": float(by_trait_a[trait][args.col_a]),
                f"{args.col_b}": float(by_trait_b[trait][args.col_b]),
            }
        )

    ra_list = [rank_a[t] for t in common_traits]
    rb_list = [rank_b[t] for t in common_traits]
    rho = spearman(ra_list, rb_list)

    by_a = sorted(rows, key=lambda r: r[f"rank_{args.label_a}"])
    by_b = sorted(rows, key=lambda r: r[f"rank_{args.label_b}"])
    by_diff = sorted(rows, key=lambda r: r["abs_rank_diff"], reverse=True)

    # Higher in a than b (rank_diff > 0)
    higher_in_a = [r for r in by_diff if r["rank_diff"] < 0]
    # Higher in b than a (rank_diff < 0)
    higher_in_b = [r for r in by_diff if r["rank_diff"] > 0]

    fieldnames = [
        "trait",
        f"rank_{args.label_a}",
        f"rank_{args.label_b}",
        "rank_diff",
        "abs_rank_diff",
        args.col_a,
        args.col_b,
    ]

    out_path = output_dir / f"{args.prefix}__comparison.csv"
    write_csv(by_diff, out_path, fieldnames=fieldnames)
    print(f"Saved comparison CSV: {out_path}")

    print(f"\nTraits compared: {len(common_traits)}")
    print(f"Spearman rank correlation ({args.label_a} vs {args.label_b}): {rho:.4f}")

    col_a_label = args.col_a[:12]
    col_b_label = args.col_b[:12]
    header = (f"{'trait':<22} {f'rank_{args.label_a}':>10} {f'rank_{args.label_b}':>10} "
              f"{'diff':>6}  {col_a_label:>12}  {col_b_label:>12}")
    sep = "-" * 80

    print(f"\nTop {args.top_k} traits ranked higher by '{args.label_a}' than '{args.label_b}':")
    print(header)
    print(sep)
    for r in higher_in_a[: args.top_k]:
        print(
            f"  {r['trait']:<20} {r[f'rank_{args.label_a}']:>10} {r[f'rank_{args.label_b}']:>10} "
            f"{r['rank_diff']:>+6}  {r[args.col_a]:>12.3f}  {r[args.col_b]:>12.3f}"
        )

    print(f"\nTop {args.top_k} traits ranked higher by '{args.label_b}' than '{args.label_a}':")
    print(header)
    print(sep)
    for r in higher_in_b[: args.top_k]:
        print(
            f"  {r['trait']:<20} {r[f'rank_{args.label_a}']:>10} {r[f'rank_{args.label_b}']:>10} "
            f"{r['rank_diff']:>+6}  {r[args.col_a]:>12.3f}  {r[args.col_b]:>12.3f}"
        )

    print(f"\nTop 10 by '{args.label_a}' ranking:")
    for r in by_a[:10]:
        print(f"  {args.label_a}={r[f'rank_{args.label_a}']:>3}  {args.label_b}={r[f'rank_{args.label_b}']:>3}  "
              f"{r['trait']:<22}  {args.col_a}={r[args.col_a]:.3f}")

    print(f"\nTop 10 by '{args.label_b}' ranking:")
    for r in by_b[:10]:
        print(f"  {args.label_b}={r[f'rank_{args.label_b}']:>3}  {args.label_a}={r[f'rank_{args.label_a}']:>3}  "
              f"{r['trait']:<22}  {args.col_b}={r[args.col_b]:.3f}")


if __name__ == "__main__":
    main()
