#!/usr/bin/env python3
"""
For each axis, plot top traits side by side for two experiments (factual vs opinion).

Produces one PNG per axis showing mean delta bars for both experiments,
so you can see at a glance which traits changed between topic types.

Usage:
  python project/plots/plot_axis_comparison.py \
    --factual-inputs  outputs/user_prompts/*/projections/strict_all_axes_llama_100eval__*.jsonl \
    --opinion-inputs  outputs/user_prompts/*/projections/opinion_axes_llama_100eval__*.jsonl \
    --output-dir      outputs/analysis/comparison/per_axis_plots \
    --top-k           6 \
    --axes            condescending entertaining emotional sycophantic universalist secular
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from plots.plot_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-axis comparison plot: factual vs opinion")
    parser.add_argument("--factual-inputs", nargs="+", required=True)
    parser.add_argument("--opinion-inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=6, help="Top traits per direction (default 6)")
    parser.add_argument("--axes", nargs="+", default=None,
                        help="Axes to plot. Default: all axes present in both experiments.")
    parser.add_argument("--factual-label", default="Factual topics")
    parser.add_argument("--opinion-label", default="Opinion topics")
    return parser.parse_args()


def infer_trait(row: dict[str, Any], path: Path) -> str | None:
    t = row.get("trait")
    if isinstance(t, str) and t.strip():
        return t.strip()
    parts = path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def load_axis_trait_stats(inputs: list[str]) -> dict[str, dict[str, dict]]:
    """Returns axis -> trait -> {mean, std, n}."""
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sum_sq: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for path_str in inputs:
        path = Path(path_str)
        if path.stem.endswith("__neutral"):
            continue
        rows = load_jsonl(path)
        for row in rows:
            axis = row.get("projection_trait")
            delta = row.get("projection_delta_trait_minus_neutral")
            if axis is None or delta is None:
                continue
            trait = infer_trait(row, path)
            if trait is None:
                continue
            axis = str(axis)
            delta = float(delta)
            sums[axis][trait] += delta
            sum_sq[axis][trait] += delta * delta
            counts[axis][trait] += 1

    result: dict[str, dict[str, dict]] = {}
    for axis, trait_sums in sums.items():
        result[axis] = {}
        for trait, s in trait_sums.items():
            n = counts[axis][trait]
            mean = s / n
            variance = (sum_sq[axis][trait] / n) - mean * mean
            std = math.sqrt(max(variance, 0.0))
            sem = std / math.sqrt(n) if n > 1 else 0.0
            result[axis][trait] = {"mean": mean, "std": std, "sem": sem, "n": n}
    return result


def top_traits(stats: dict[str, dict], k: int) -> tuple[list[str], list[str]]:
    """Return top-k positive and top-k negative traits by mean delta."""
    sorted_traits = sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    pos = [t for t, _ in sorted_traits if sorted_traits[list(t for t, _ in sorted_traits).index(t)][1]["mean"] > 0][:k]
    neg = [t for t, _ in reversed(sorted_traits) if sorted_traits[list(t for t, _ in sorted_traits).index(t)][1]["mean"] < 0][:k]
    # simpler version
    by_mean = sorted(stats.items(), key=lambda x: x[1]["mean"])
    pos = [t for t, v in by_mean if v["mean"] > 0][-k:][::-1]
    neg = [t for t, v in by_mean if v["mean"] < 0][:k]
    return pos, neg


def plot_axis(
    axis: str,
    fact_stats: dict[str, dict],
    opin_stats: dict[str, dict],
    top_k: int,
    fact_label: str,
    opin_label: str,
    out_path: Path,
) -> None:
    # Union of top traits from both experiments
    fact_pos, fact_neg = top_traits(fact_stats, top_k)
    opin_pos, opin_neg = top_traits(opin_stats, top_k)

    pos_traits = list(dict.fromkeys(fact_pos + opin_pos))[:top_k + 2]
    neg_traits = list(dict.fromkeys(fact_neg + opin_neg))[:top_k + 2]
    all_traits = pos_traits + neg_traits[::-1]

    if not all_traits:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_traits) * 0.55 + 1.5)))

    y_positions = range(len(all_traits))
    bar_height = 0.35
    color_fact = "#4C72B0"
    color_opin = "#DD8452"

    for i, trait in enumerate(all_traits):
        y = i
        # factual bar
        if trait in fact_stats:
            v = fact_stats[trait]
            ax.barh(y + bar_height / 2, v["mean"], height=bar_height,
                    color=color_fact, alpha=0.85,
                    xerr=v["sem"], error_kw={"elinewidth": 1, "capsize": 2})
        # opinion bar
        if trait in opin_stats:
            v = opin_stats[trait]
            ax.barh(y - bar_height / 2, v["mean"], height=bar_height,
                    color=color_opin, alpha=0.85,
                    xerr=v["sem"], error_kw={"elinewidth": 1, "capsize": 2})

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(all_traits, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean projection delta (trait − neutral)", fontsize=9)
    ax.set_title(f"Axis: {axis}", fontsize=11, fontweight="bold")

    fact_patch = mpatches.Patch(color=color_fact, alpha=0.85, label=fact_label)
    opin_patch = mpatches.Patch(color=color_opin, alpha=0.85, label=opin_label)
    ax.legend(handles=[fact_patch, opin_patch], fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading factual projections...")
    fact_data = load_axis_trait_stats(args.factual_inputs)
    print("Loading opinion projections...")
    opin_data = load_axis_trait_stats(args.opinion_inputs)

    axes_to_plot = args.axes if args.axes else sorted(set(fact_data) & set(opin_data))
    print(f"Plotting {len(axes_to_plot)} axes...")

    for axis in axes_to_plot:
        if axis not in fact_data or axis not in opin_data:
            print(f"  Skipping {axis} (missing in one experiment)")
            continue
        out_path = out_dir / f"{axis}.png"
        plot_axis(
            axis=axis,
            fact_stats=fact_data[axis],
            opin_stats=opin_data[axis],
            top_k=args.top_k,
            fact_label=args.factual_label,
            opin_label=args.opinion_label,
            out_path=out_path,
        )
        print(f"  Saved {out_path}")

    print(f"\nDone. {len(axes_to_plot)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
