#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import infer_run_label, load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting many traits across many axes as grouped bars."""
    parser = argparse.ArgumentParser(
        description="Plot many user-trait runs across many projection axes as grouped bars with variance."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Projection JSONL files from multiple trait runs",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional labels corresponding to --inputs. Defaults to inferred trait labels.",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Also include the aggregated neutral baseline from projection_score_neutral.",
    )
    parser.add_argument(
        "--neutral-label",
        type=str,
        default="Neutral",
        help="Display label for the aggregated neutral baseline.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV summary output path",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=None,
        help="Optional: keep only top K axes by average absolute trait mean across runs",
    )
    parser.add_argument(
        "--axis-filter",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of axes to keep.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional batch size for variance estimation. When set, scores are first averaged "
            "within contiguous example batches and std is computed across those batch means."
        ),
    )
    parser.add_argument(
        "--drop-incomplete-batch",
        action="store_true",
        help="When batching is enabled, drop the final partial batch instead of keeping it.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()


def compute_stats(values: list[float]) -> dict[str, float | int]:
    """Compute count, mean, std, min, and max for one axis/condition group."""
    if not values:
        raise ValueError("Cannot summarize empty values")

    count = len(values)
    mean = sum(values) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in values) / (count - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    return {
        "count": count,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def collect_axis_values(rows: list[dict[str, Any]], value_key: str) -> dict[str, list[float]]:
    """Collect projection values grouped by axis from one projection file."""
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis_trait = row.get("projection_trait")
        if axis_trait is None:
            continue
        by_axis.setdefault(str(axis_trait), []).append(float(row[value_key]))
    return by_axis


def make_example_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Build a stable per-example key so repeated axis rows can be regrouped into original examples."""
    return (
        row.get("intent_index"),
        row.get("candidate_index"),
        row.get("intent"),
        row.get("neutral_prompt"),
        row.get("trait_prompt"),
    )


def collect_batched_axis_values(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    batch_size: int,
    drop_incomplete_batch: bool,
) -> dict[str, list[float]]:
    """Collect per-axis batch means instead of per-example values."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    example_order: list[tuple[Any, ...]] = []
    example_seen: set[tuple[Any, ...]] = set()
    by_axis_example: dict[str, dict[tuple[Any, ...], float]] = {}

    for row in rows:
        axis_trait = row.get("projection_trait")
        if axis_trait is None:
            continue
        axis_trait = str(axis_trait)
        example_key = make_example_key(row)
        if example_key not in example_seen:
            example_seen.add(example_key)
            example_order.append(example_key)
        by_axis_example.setdefault(axis_trait, {})[example_key] = float(row[value_key])

    batched: dict[str, list[float]] = {}
    for axis_trait, example_scores in by_axis_example.items():
        ordered_values = [example_scores[key] for key in example_order if key in example_scores]
        if not ordered_values:
            continue

        batch_means: list[float] = []
        for start in range(0, len(ordered_values), batch_size):
            batch = ordered_values[start : start + batch_size]
            if len(batch) < batch_size and drop_incomplete_batch:
                continue
            batch_means.append(sum(batch) / len(batch))
        batched[axis_trait] = batch_means

    return batched


def axis_intersection(series: list[dict[str, list[float]]]) -> list[str]:
    """Return sorted axes that are present in every provided series."""
    if not series:
        return []
    shared = set(series[0].keys())
    for item in series[1:]:
        shared &= set(item.keys())
    return sorted(shared)


def main() -> None:
    """Aggregate many-axis projections across runs and render a grouped bar chart with error bars."""
    args = parse_args()

    if args.labels is not None and len(args.labels) != len(args.inputs):
        raise ValueError("--labels must have the same length as --inputs")

    trait_series: list[tuple[str, dict[str, list[float]]]] = []
    neutral_inputs: list[dict[str, list[float]]] = []

    for idx, input_path_str in enumerate(args.inputs):
        input_path = Path(input_path_str)
        label = args.labels[idx] if args.labels is not None else infer_run_label(input_path)
        rows = load_jsonl(input_path)
        if not rows:
            raise ValueError(f"No projection rows found in {input_path}")

        if args.batch_size is not None:
            trait_by_axis = collect_batched_axis_values(
                rows,
                "projection_score_trait",
                batch_size=args.batch_size,
                drop_incomplete_batch=args.drop_incomplete_batch,
            )
        else:
            trait_by_axis = collect_axis_values(rows, "projection_score_trait")
        trait_series.append((label, trait_by_axis))

        if args.include_neutral:
            if args.batch_size is not None:
                neutral_inputs.append(
                    collect_batched_axis_values(
                        rows,
                        "projection_score_neutral",
                        batch_size=args.batch_size,
                        drop_incomplete_batch=args.drop_incomplete_batch,
                    )
                )
            else:
                neutral_inputs.append(collect_axis_values(rows, "projection_score_neutral"))

    all_series = [series for _, series in trait_series]
    merged_neutral: dict[str, list[float]] = {}
    if args.include_neutral:
        for series in neutral_inputs:
            for axis_trait, values in series.items():
                merged_neutral.setdefault(axis_trait, []).extend(values)
        all_series = [merged_neutral, *all_series]

    if args.axis_filter:
        axis_traits = list(dict.fromkeys(args.axis_filter))
    else:
        axis_traits = axis_intersection(all_series)

    if not axis_traits:
        raise ValueError("No shared axes found across inputs")

    if args.top_k_axes is not None and args.axis_filter is None:
        axis_strength: dict[str, list[float]] = {}
        for _, series in trait_series:
            for axis_trait, values in series.items():
                axis_strength.setdefault(axis_trait, []).append(abs(compute_stats(values)["mean"]))
        ranked_axes = sorted(
            axis_traits,
            key=lambda axis: sum(axis_strength.get(axis, [])) / max(len(axis_strength.get(axis, [])), 1),
            reverse=True,
        )
        axis_traits = ranked_axes[: args.top_k_axes]

    summary_rows: list[dict[str, Any]] = []
    plot_series: list[tuple[str, dict[str, dict[str, float | int]]]] = []

    if args.include_neutral:
        neutral_summary: dict[str, dict[str, float | int]] = {}
        for axis_trait in axis_traits:
            neutral_summary[axis_trait] = compute_stats(merged_neutral[axis_trait])
            summary_rows.append(
                {
                    "run_label": args.neutral_label,
                    "axis_trait": axis_trait,
                    "batched": args.batch_size is not None,
                    "batch_size": args.batch_size,
                    **neutral_summary[axis_trait],
                }
            )
        plot_series.append((args.neutral_label, neutral_summary))

    for label, series in trait_series:
        series_summary: dict[str, dict[str, float | int]] = {}
        for axis_trait in axis_traits:
            series_summary[axis_trait] = compute_stats(series[axis_trait])
            summary_rows.append(
                {
                    "run_label": label,
                    "axis_trait": axis_trait,
                    "batched": args.batch_size is not None,
                    "batch_size": args.batch_size,
                    **series_summary[axis_trait],
                }
            )
        plot_series.append((label, series_summary))

    num_axes = len(axis_traits)
    num_series = len(plot_series)
    x = np.arange(num_axes)
    group_width = 0.82
    bar_width = group_width / max(num_series, 1)
    offsets = (np.arange(num_series) - (num_series - 1) / 2.0) * bar_width

    plt.figure(figsize=(max(12, num_axes * 1.3), 7))
    cmap = plt.get_cmap("tab10")

    for idx, (label, series_summary) in enumerate(plot_series):
        means = [float(series_summary[axis]["mean"]) for axis in axis_traits]
        stds = [float(series_summary[axis]["std"]) for axis in axis_traits]
        positions = x + offsets[idx]
        plt.bar(
            positions,
            means,
            width=bar_width * 0.92,
            label=label,
            color=cmap(idx % 10),
            alpha=0.9,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.2, "alpha": 0.9},
        )

    plt.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    plt.xticks(x, axis_traits, rotation=35, ha="right")
    ylabel = "Mean projection score"
    if args.batch_size is not None:
        ylabel += f" (batch means, batch_size={args.batch_size})"
    plt.ylabel(ylabel)
    plt.xlabel("Projection axis")
    plt.title(args.title or "Many traits across many axes")
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        write_csv(
            summary_rows,
            Path(args.csv_output),
            fieldnames=["run_label", "axis_trait", "batched", "batch_size", "count", "mean", "std", "min", "max"],
        )

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved CSV summary to {args.csv_output}")


if __name__ == "__main__":
    main()
