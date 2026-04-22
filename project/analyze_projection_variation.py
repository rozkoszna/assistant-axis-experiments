#!/usr/bin/env python3
"""
Summarize variation from precomputed projection JSONL files.

This script reads existing projection outputs and computes per-axis summary
statistics for each condition, such as:
- count
- mean
- std
- min
- max

Typical use:
- compare `confused` vs `very_confused`
- include the matched neutral baseline from the same files
- inspect variation without rerunning generation or projection
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for path in (REPO_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from plots.plot_utils import load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for projection-summary analysis."""
    parser = argparse.ArgumentParser(
        description="Compute mean/std/count summaries from precomputed projection JSONL files."
    )
    parser.add_argument(
        "--trait-inputs",
        nargs="+",
        required=True,
        help="Projection JSONL files for trait-conditioned runs.",
    )
    parser.add_argument(
        "--trait-labels",
        nargs="+",
        required=True,
        help="Display labels corresponding to --trait-inputs.",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Also summarize projection_score_neutral aggregated across all trait inputs.",
    )
    parser.add_argument(
        "--neutral-label",
        type=str,
        default="Neutral",
        help="Label to use for the aggregated neutral condition.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON output path for the full summary.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path for the flat summary table.",
    )
    parser.add_argument(
        "--axis-filter",
        nargs="*",
        default=None,
        help="Optional list of axis traits to keep. Default: keep all axes present in every condition.",
    )
    return parser.parse_args()


def compute_stats(values: list[float]) -> dict[str, float | int]:
    """Compute basic summary statistics for one axis/condition slice."""
    if not values:
        raise ValueError("Cannot summarize an empty list of values")

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
    """Collect all projection values grouped by axis from one projection file."""
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue
        by_axis.setdefault(str(axis), []).append(float(row[value_key]))
    return by_axis


def axis_intersection(series: list[dict[str, list[float]]]) -> list[str]:
    """Return sorted axes that are present in every provided series."""
    if not series:
        return []
    shared = set(series[0].keys())
    for item in series[1:]:
        shared &= set(item.keys())
    return sorted(shared)


def main() -> None:
    """Load projection files, compute per-axis summaries, and write optional outputs."""
    args = parse_args()

    if len(args.trait_inputs) != len(args.trait_labels):
        raise ValueError("--trait-inputs and --trait-labels must have the same length")

    trait_series: list[tuple[str, dict[str, list[float]]]] = []
    neutral_series_inputs: list[dict[str, list[float]]] = []

    for input_path, label in zip(args.trait_inputs, args.trait_labels):
        rows = load_jsonl(Path(input_path))
        if not rows:
            raise ValueError(f"No rows found in trait input: {input_path}")
        trait_by_axis = collect_axis_values(rows, "projection_score_trait")
        trait_series.append((label, trait_by_axis))
        if args.include_neutral:
            neutral_series_inputs.append(collect_axis_values(rows, "projection_score_neutral"))

    all_series_dicts = [series for _, series in trait_series]
    if args.include_neutral:
        merged_neutral: dict[str, list[float]] = {}
        for series in neutral_series_inputs:
            for axis, values in series.items():
                merged_neutral.setdefault(axis, []).extend(values)
        all_series_dicts = [merged_neutral, *all_series_dicts]
    else:
        merged_neutral = {}

    if args.axis_filter:
        axes = sorted(set(args.axis_filter))
    else:
        axes = axis_intersection(all_series_dicts)

    if not axes:
        raise ValueError("No shared axes found across the provided inputs")

    summary_rows: list[dict[str, Any]] = []

    if args.include_neutral:
        for axis in axes:
            stats = compute_stats(merged_neutral[axis])
            summary_rows.append(
                {
                    "condition": args.neutral_label,
                    "axis": axis,
                    **stats,
                }
            )

    for label, series in trait_series:
        for axis in axes:
            stats = compute_stats(series[axis])
            summary_rows.append(
                {
                    "condition": label,
                    "axis": axis,
                    **stats,
                }
            )

    print("\nVariation summary")
    print("=================")
    for row in summary_rows:
        print(
            f"{row['condition']:>15} | {row['axis']:<16} "
            f"n={row['count']:<3} mean={row['mean']:.4f} std={row['std']:.4f} "
            f"min={row['min']:.4f} max={row['max']:.4f}"
        )

    if args.output_csv:
        write_csv(
            summary_rows,
            Path(args.output_csv),
            fieldnames=["condition", "axis", "count", "mean", "std", "min", "max"],
        )
        print(f"\nSaved CSV summary to {args.output_csv}")

    if args.output_json:
        payload = {
            "axes": axes,
            "rows": summary_rows,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
