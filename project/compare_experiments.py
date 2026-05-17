#!/usr/bin/env python3
"""
Compare two significance experiments side by side.

Typical use:
  python project/compare_experiments.py \
    --factual-dir  outputs/analysis/significance \
    --opinion-dir  outputs/analysis/opinion/significance \
    --factual-prefix  strict_all_axes \
    --opinion-prefix  opinion_axes \
    --output-dir  outputs/analysis/comparison
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two significance experiments")
    parser.add_argument("--factual-dir", required=True)
    parser.add_argument("--opinion-dir", required=True)
    parser.add_argument("--factual-prefix", default="strict_all_axes")
    parser.add_argument("--opinion-prefix", default="opinion_axes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v, fmt_str=".3f"):
    try:
        return format(float(v), fmt_str)
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    args = parse_args()
    factual_dir = Path(args.factual_dir)
    opinion_dir = Path(args.opinion_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load trait summaries
    f_traits = {r["trait"]: r for r in load_csv(factual_dir / f"{args.factual_prefix}__trait_summary.csv")}
    o_traits = {r["trait"]: r for r in load_csv(opinion_dir / f"{args.opinion_prefix}__trait_summary.csv")}

    # Load axis summaries
    f_axes = {r["axis"]: r for r in load_csv(factual_dir / f"{args.factual_prefix}__axis_summary.csv")}
    o_axes = {r["axis"]: r for r in load_csv(opinion_dir / f"{args.opinion_prefix}__axis_summary.csv")}

    # Load pair tables
    f_pairs = load_csv(factual_dir / f"{args.factual_prefix}__trait_axis_pairs.csv")
    o_pairs = load_csv(opinion_dir / f"{args.opinion_prefix}__trait_axis_pairs.csv")

    common_traits = sorted(set(f_traits) & set(o_traits))
    common_axes   = sorted(set(f_axes)   & set(o_axes))

    # ------------------------------------------------------------------ #
    # 1. Trait comparison table
    # ------------------------------------------------------------------ #
    trait_rows = []
    for trait in common_traits:
        f = f_traits[trait]
        o = o_traits[trait]
        f_sig = int(f["n_significant_axes"])
        o_sig = int(o["n_significant_axes"])
        f_d   = float(f["mean_abs_cohen_d"])
        o_d   = float(o["mean_abs_cohen_d"])
        f_exc = float(f["mean_exceedance_rate"]) if f.get("mean_exceedance_rate") not in ("", "nan") else float("nan")
        o_exc = float(o["mean_exceedance_rate"]) if o.get("mean_exceedance_rate") not in ("", "nan") else float("nan")
        trait_rows.append({
            "trait": trait,
            "factual_n_sig": f_sig,
            "opinion_n_sig": o_sig,
            "delta_n_sig": o_sig - f_sig,
            "factual_mean_d": f_d,
            "opinion_mean_d": o_d,
            "delta_mean_d": o_d - f_d,
            "factual_exceedance": f_exc,
            "opinion_exceedance": o_exc,
            "delta_exceedance": (o_exc - f_exc) if not (math.isnan(f_exc) or math.isnan(o_exc)) else float("nan"),
        })

    trait_rows.sort(key=lambda r: r["delta_n_sig"], reverse=True)
    write_csv(
        trait_rows, out_dir / "trait_comparison.csv",
        ["trait", "factual_n_sig", "opinion_n_sig", "delta_n_sig",
         "factual_mean_d", "opinion_mean_d", "delta_mean_d",
         "factual_exceedance", "opinion_exceedance", "delta_exceedance"],
    )

    # ------------------------------------------------------------------ #
    # 2. Axis comparison table
    # ------------------------------------------------------------------ #
    axis_rows = []
    for axis in common_axes:
        f = f_axes[axis]
        o = o_axes[axis]
        f_sig = int(f["n_significant_traits"])
        o_sig = int(o["n_significant_traits"])
        f_d   = float(f["mean_abs_cohen_d"])
        o_d   = float(o["mean_abs_cohen_d"])
        f_exc = float(f["mean_exceedance_rate"]) if f.get("mean_exceedance_rate") not in ("", "nan") else float("nan")
        o_exc = float(o["mean_exceedance_rate"]) if o.get("mean_exceedance_rate") not in ("", "nan") else float("nan")
        axis_rows.append({
            "axis": axis,
            "factual_n_sig": f_sig,
            "opinion_n_sig": o_sig,
            "delta_n_sig": o_sig - f_sig,
            "factual_mean_d": f_d,
            "opinion_mean_d": o_d,
            "delta_mean_d": o_d - f_d,
            "factual_exceedance": f_exc,
            "opinion_exceedance": o_exc,
            "delta_exceedance": (o_exc - f_exc) if not (math.isnan(f_exc) or math.isnan(o_exc)) else float("nan"),
        })

    axis_rows.sort(key=lambda r: r["delta_n_sig"], reverse=True)
    write_csv(
        axis_rows, out_dir / "axis_comparison.csv",
        ["axis", "factual_n_sig", "opinion_n_sig", "delta_n_sig",
         "factual_mean_d", "opinion_mean_d", "delta_mean_d",
         "factual_exceedance", "opinion_exceedance", "delta_exceedance"],
    )

    # ------------------------------------------------------------------ #
    # 3. Pairs significant in opinion but NOT in factual (new effects)
    # ------------------------------------------------------------------ #
    f_sig_pairs = {(r["trait"], r["axis"]) for r in f_pairs if r["significant"] == "True"}
    o_sig_pairs = {(r["trait"], r["axis"]) for r in o_pairs if r["significant"] == "True"}

    new_in_opinion = sorted(o_sig_pairs - f_sig_pairs)
    lost_in_opinion = sorted(f_sig_pairs - o_sig_pairs)

    o_pairs_dict = {(r["trait"], r["axis"]): r for r in o_pairs}
    f_pairs_dict = {(r["trait"], r["axis"]): r for r in f_pairs}

    new_rows = [
        {"trait": t, "axis": a,
         "opinion_cohen_d": float(o_pairs_dict[(t, a)]["cohen_d"]),
         "opinion_exceedance": float(o_pairs_dict[(t, a)].get("exceedance_rate", "nan") or "nan")}
        for t, a in new_in_opinion
        if (t, a) in o_pairs_dict
    ]
    new_rows.sort(key=lambda r: abs(r["opinion_cohen_d"]), reverse=True)
    write_csv(new_rows, out_dir / "new_significant_in_opinion.csv",
              ["trait", "axis", "opinion_cohen_d", "opinion_exceedance"])

    lost_rows = [
        {"trait": t, "axis": a,
         "factual_cohen_d": float(f_pairs_dict[(t, a)]["cohen_d"]),
         "factual_exceedance": float(f_pairs_dict[(t, a)].get("exceedance_rate", "nan") or "nan")}
        for t, a in lost_in_opinion
        if (t, a) in f_pairs_dict
    ]
    lost_rows.sort(key=lambda r: abs(r["factual_cohen_d"]), reverse=True)
    write_csv(lost_rows, out_dir / "lost_significant_in_opinion.csv",
              ["trait", "axis", "factual_cohen_d", "factual_exceedance"])

    # ------------------------------------------------------------------ #
    # 4. Print summary
    # ------------------------------------------------------------------ #
    f_total_sig = sum(1 for r in f_pairs if r["significant"] == "True")
    o_total_sig = sum(1 for r in o_pairs if r["significant"] == "True")

    print(f"\n{'='*65}")
    print("EXPERIMENT COMPARISON: factual vs opinion intents")
    print(f"{'='*65}")
    print(f"  Significant pairs — factual: {f_total_sig}   opinion: {o_total_sig}   delta: {o_total_sig - f_total_sig:+d}")
    print(f"  New significant in opinion (not in factual): {len(new_in_opinion)}")
    print(f"  Lost significant in opinion (was in factual): {len(lost_in_opinion)}")

    print(f"\nTop {args.top_k} traits where opinion > factual (more axes move):")
    print(f"  {'trait':<22} {'fact_sig':>8} {'opin_sig':>8} {'Δsig':>6}  {'fact_d':>7}  {'opin_d':>7}  {'Δd':>7}")
    print("  " + "-" * 68)
    for r in trait_rows[:args.top_k]:
        print(f"  {r['trait']:<22} {r['factual_n_sig']:>8} {r['opinion_n_sig']:>8} "
              f"{r['delta_n_sig']:>+6}  {r['factual_mean_d']:>7.3f}  {r['opinion_mean_d']:>7.3f}  "
              f"{r['delta_mean_d']:>+7.3f}")

    print(f"\nTop {args.top_k} traits where factual > opinion (fewer axes move):")
    print(f"  {'trait':<22} {'fact_sig':>8} {'opin_sig':>8} {'Δsig':>6}  {'fact_d':>7}  {'opin_d':>7}  {'Δd':>7}")
    print("  " + "-" * 68)
    for r in trait_rows[-args.top_k:][::-1]:
        print(f"  {r['trait']:<22} {r['factual_n_sig']:>8} {r['opinion_n_sig']:>8} "
              f"{r['delta_n_sig']:>+6}  {r['factual_mean_d']:>7.3f}  {r['opinion_mean_d']:>7.3f}  "
              f"{r['delta_mean_d']:>+7.3f}")

    print(f"\nTop {args.top_k} axes unlocked by opinion topics (more traits move them):")
    print(f"  {'axis':<22} {'fact_sig':>8} {'opin_sig':>8} {'Δsig':>6}  {'fact_d':>7}  {'opin_d':>7}")
    print("  " + "-" * 62)
    for r in axis_rows[:args.top_k]:
        print(f"  {r['axis']:<22} {r['factual_n_sig']:>8} {r['opinion_n_sig']:>8} "
              f"{r['delta_n_sig']:>+6}  {r['factual_mean_d']:>7.3f}  {r['opinion_mean_d']:>7.3f}")

    print(f"\nTop {args.top_k} axes suppressed by opinion topics (fewer traits move them):")
    print(f"  {'axis':<22} {'fact_sig':>8} {'opin_sig':>8} {'Δsig':>6}  {'fact_d':>7}  {'opin_d':>7}")
    print("  " + "-" * 62)
    for r in axis_rows[-args.top_k:][::-1]:
        print(f"  {r['axis']:<22} {r['factual_n_sig']:>8} {r['opinion_n_sig']:>8} "
              f"{r['delta_n_sig']:>+6}  {r['factual_mean_d']:>7.3f}  {r['opinion_mean_d']:>7.3f}")

    print(f"\nTop 10 new significant pairs in opinion (by |Cohen's d|):")
    for r in new_rows[:10]:
        print(f"  {r['trait']:<22} {r['axis']:<22} d={r['opinion_cohen_d']:+.3f}")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
