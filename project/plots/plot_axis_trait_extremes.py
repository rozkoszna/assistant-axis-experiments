#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import sys

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
            "How to rank traits before taking top-k extremes: "
            "mean_delta (trait-neutral mean shift), "
            "variance_gap (trait variance - neutral variance), "
            "std_gap (trait std - neutral std), "
            "global_delta (trait mean score - one global neutral mean for the axis)."
        ),
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for per-axis PNGs")
    return parser.parse_args()


def infer_trait(row: dict[str, Any], input_path: Path) -> str | None:
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
        # One neutral baseline per axis, aggregated across all examples/traits.
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
                    "mean_delta": delta_stats["mean"],
                    "std_delta": delta_stats["std"],
                    "variance_delta": delta_stats["variance"],
                    "mean_trait_score": trait_stats["mean"],
                    "std_trait_score": trait_stats["std"],
                    "mean_neutral_score": neutral_stats["mean"],
                    "std_neutral_score": neutral_stats["std"],
                    "variance_neutral_score": neutral_stats["variance"],
                    "variance_gap": trait_stats["variance"] - neutral_stats["variance"],
                    "std_gap": trait_stats["std"] - neutral_stats["std"],
                    "global_delta": trait_stats["mean"] - neutral_mean,
                }
            )
        if not stats_rows:
            continue

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
