#!/usr/bin/env python3
"""
For each personality axis, find the user traits that produce the most extreme
projection scores (most positive and most negative), and plot them as a bar chart
against a neutral baseline.

One PNG is saved per axis to --output-dir. This is useful for a quick sanity check:
if a trait pushes the model strongly in the expected direction on a given axis,
the axis and the trait list are working as intended.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import sys

# Allow importing from project/plots/plot_utils regardless of where the script is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for search_path in (REPO_ROOT, PROJECT_ROOT):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from plots.plot_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each axis, pick top positive/negative user traits and plot "
            "trait means with std error bars, plus one aggregated Neutral baseline."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Projection JSONL files")
    parser.add_argument("--top-k", type=int, default=4, help="Top traits per direction per axis")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum rows for trait-axis stats")
    parser.add_argument(
        "--rank-by",
        choices=["mean_delta", "variance_gap", "std_gap", "global_delta"],
        default="mean_delta",
        help=(
            "How to rank traits before taking top-k extremes. "
            "mean_delta: per-trait (score_trait - score_neutral), captures average shift caused by the trait. "
            "global_delta: trait mean - one shared neutral mean for the whole axis, ignores per-example pairing. "
            "variance_gap / std_gap: how much more variable trait responses are vs neutral — useful for detecting "
            "traits that destabilise the model rather than shift it in one direction."
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for per-axis PNGs")
    return parser.parse_args()


def infer_trait(row: dict[str, Any], input_path: Path) -> str | None:
    """Return the user trait label for a projection row.

    Newer output files store the trait directly in the row. Older outputs
    encoded the trait in the directory structure (…/user_prompts/<trait>/…),
    so we fall back to path parsing when the field is missing.
    """
    trait = row.get("trait")
    if isinstance(trait, str) and trait.strip():
        return trait.strip()
    parts = input_path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def compute_stats(values: list[float]) -> dict[str, float]:
    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(variance)
    else:
        variance = 0.0
        std = 0.0
    return {"mean": mean, "std": std, "variance": variance}


def sanitize_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return safe.strip("_") or "axis"


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    by_axis_trait_delta: dict[str, dict[str, list[float]]] = {}
    by_axis_trait_trait_score: dict[str, dict[str, list[float]]] = {}
    by_axis_trait_neutral_score: dict[str, dict[str, list[float]]] = {}
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        rows = load_jsonl(input_path)
        for row in rows:
            axis = row.get("projection_trait")
            delta = row.get("projection_delta_trait_minus_neutral")
            trait_score = row.get("projection_score_trait")
            neutral_score = row.get("projection_score_neutral")
            if axis is None or delta is None or trait_score is None or neutral_score is None:
                continue
            trait = infer_trait(row, input_path)
            if trait is None:
                continue
            axis_key = str(axis)
            by_axis_trait_delta.setdefault(axis_key, {}).setdefault(trait, []).append(float(delta))
            by_axis_trait_trait_score.setdefault(axis_key, {}).setdefault(trait, []).append(float(trait_score))
            by_axis_trait_neutral_score.setdefault(axis_key, {}).setdefault(trait, []).append(float(neutral_score))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for axis in sorted(by_axis_trait_delta.keys()):
        stats_rows: list[dict[str, Any]] = []

        # Neutral baseline: pool all neutral scores across every trait for this axis.
        # A single shared baseline makes it easy to see whether any trait actually
        # shifts the model — bars above/below the dashed line are meaningful.
        all_neutral_values: list[float] = []
        for neutral_values in by_axis_trait_neutral_score[axis].values():
            all_neutral_values.extend(neutral_values)
        neutral_stats = compute_stats(all_neutral_values)
        neutral_mean = neutral_stats["mean"]
        neutral_std = neutral_stats["std"]

        for trait, values in by_axis_trait_delta[axis].items():
            if len(values) < args.min_count:
                continue
            delta_stats = compute_stats(values)
            trait_stats = compute_stats(by_axis_trait_trait_score[axis][trait])
            neutral_stats = compute_stats(by_axis_trait_neutral_score[axis][trait])
            stats_rows.append(
                {
                    "trait": trait,
                    "count": len(values),
                    # Per-example delta (trait score − its paired neutral score).
                    "mean_delta": delta_stats["mean"],
                    "std_delta": delta_stats["std"],
                    "variance_delta": delta_stats["variance"],
                    "mean_trait_score": trait_stats["mean"],
                    "std_trait_score": trait_stats["std"],
                    "mean_neutral_score": neutral_stats["mean"],
                    "std_neutral_score": neutral_stats["std"],
                    "variance_neutral_score": neutral_stats["variance"],
                    # How much more spread out trait responses are vs neutral responses.
                    "variance_gap": trait_stats["variance"] - neutral_stats["variance"],
                    "std_gap": trait_stats["std"] - neutral_stats["std"],
                    # Unpaired: trait mean vs the single pooled neutral mean for this axis.
                    "global_delta": trait_stats["mean"] - neutral_mean,
                }
            )
        if not stats_rows:
            continue

        # Select the top-K most positive and top-K most negative traits by the chosen metric,
        # then lay them out left-to-right (negative → positive) so the plot reads intuitively.
        rank_key = args.rank_by
        pos = sorted(stats_rows, key=lambda item: item[rank_key], reverse=True)[: args.top_k]
        neg = sorted(stats_rows, key=lambda item: item[rank_key])[: args.top_k]
        selected = neg + pos
        if not selected:
            continue

        labels = [item["trait"] for item in selected]
        trait_means = [item["mean_trait_score"] for item in selected]
        trait_stds = [item["std_trait_score"] for item in selected]

        width = max(10, 0.9 * len(selected))
        fig, ax = plt.subplots(figsize=(width, 5))
        x = list(range(len(labels)))
        # Red = traits that push the model toward the negative pole of the axis,
        # green = traits that push it toward the positive pole.
        trait_colors = ["#d95f02"] * len(neg) + ["#1b9e77"] * len(pos)
        ax.bar(
            x,
            trait_means,
            yerr=trait_stds,
            capsize=3,
            width=0.72,
            color=trait_colors,
            alpha=0.9,
            label="Trait score",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Mean projection score")

        # Dashed line = pooled neutral mean; shaded band = ±1 std of neutral responses.
        # Bars outside the band indicate a meaningful shift caused by the user trait.
        ax.axhline(
            neutral_mean,
            color="#4e79a7",
            linewidth=2,
            linestyle="--",
            label=f"Neutral baseline (mean={neutral_mean:.3f})",
        )
        ax.axhspan(
            neutral_mean - neutral_std,
            neutral_mean + neutral_std,
            color="#4e79a7",
            alpha=0.12,
            label=f"Neutral ±1 std ({neutral_std:.3f})",
        )

        ax.set_title(
            f"Axis: {axis} | top -{args.top_k}/+{args.top_k} by {args.rank_by}"
        )
        ax.legend(loc="best")
        fig.tight_layout()

        output_path = output_dir / f"{sanitize_name(axis)}__top{args.top_k}_extremes.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
