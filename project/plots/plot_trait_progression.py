#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot progression across projection axes for a baseline run and one or more trait runs."
    )
    parser.add_argument(
        "--baseline-input",
        type=str,
        required=True,
        help="Projection JSONL used for the baseline series (usually take neutral scores from this file).",
    )
    parser.add_argument(
        "--trait-inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more projection JSONL files for trait runs.",
    )
    parser.add_argument(
        "--trait-labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for trait-inputs, same length and order.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="neutral",
        help="Label for the baseline series.",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top-k-axes", type=int, default=20)
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Trait progression across axes",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")

    if mode == "mean":
        return sum(values) / len(values)

    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def summarize_by_axis(rows: list[dict[str, Any]], aggregate_mode: str) -> dict[str, dict[str, float]]:
    by_axis: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        axis = row["projection_trait"]
        by_axis.setdefault(
            axis,
            {
                "neutral": [],
                "trait": [],
                "delta": [],
            },
        )
        by_axis[axis]["neutral"].append(float(row["projection_score_neutral"]))
        by_axis[axis]["trait"].append(float(row["projection_score_trait"]))
        by_axis[axis]["delta"].append(float(row["projection_delta_neutral_minus_trait"]))

    out: dict[str, dict[str, float]] = {}
    for axis, vals in by_axis.items():
        out[axis] = {
            "neutral": aggregate(vals["neutral"], aggregate_mode),
            "trait": aggregate(vals["trait"], aggregate_mode),
            "delta": aggregate(vals["delta"], aggregate_mode),
        }
    return out


def main() -> None:
    args = parse_args()

    if len(args.trait_inputs) != len(args.trait_labels):
        raise ValueError("--trait-inputs and --trait-labels must have the same length")

    baseline_rows = read_jsonl(Path(args.baseline_input))
    if not baseline_rows:
        raise ValueError(f"No rows found in {args.baseline_input}")

    baseline_summary = summarize_by_axis(baseline_rows, args.aggregate)

    trait_summaries: list[tuple[str, dict[str, dict[str, float]]]] = []
    for label, input_path in zip(args.trait_labels, args.trait_inputs):
        rows = read_jsonl(Path(input_path))
        if not rows:
            raise ValueError(f"No rows found in {input_path}")
        trait_summaries.append((label, summarize_by_axis(rows, args.aggregate)))

    common_axes = set(baseline_summary.keys())
    for _, summary in trait_summaries:
        common_axes &= set(summary.keys())

    common_axes = sorted(common_axes)
    if not common_axes:
        raise ValueError("No common projection axes found across all inputs")

    def axis_strength(axis: str) -> float:
        total = abs(baseline_summary[axis]["delta"])
        for _, summary in trait_summaries:
            total += abs(summary[axis]["delta"])
        return total

    ranked_axes = sorted(common_axes, key=axis_strength, reverse=True)
    axes = ranked_axes[: args.top_k_axes]

    x = list(range(len(axes)))

    plt.figure(figsize=(max(12, 0.7 * len(axes)), 7))

    baseline_vals = [baseline_summary[axis]["neutral"] for axis in axes]

    trait_values_by_label: list[tuple[str, list[float]]] = []
    for label, summary in trait_summaries:
        vals = [summary[axis]["trait"] for axis in axes]
        trait_values_by_label.append((label, vals))

    for i in range(len(axes)):
        series_y = [baseline_vals[i]] + [vals[i] for _, vals in trait_values_by_label]
        series_x = [x[i]] * len(series_y)
        plt.plot(series_x, series_y, alpha=0.5)

    plt.scatter(x, baseline_vals, label=args.baseline_label)

    for label, vals in trait_values_by_label:
        plt.scatter(x, vals, label=label)

    plt.xticks(x, axes, rotation=45, ha="right")
    plt.ylabel("Projection score")
    plt.xlabel("Axis")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()