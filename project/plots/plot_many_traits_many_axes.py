#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot many user-trait runs across many projection axes."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Projection JSONL files from multiple trait runs",
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
        "--metric",
        choices=["delta", "neutral", "trait"],
        default="delta",
        help="Which value to aggregate per run/axis",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation over rows within each run/axis",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=None,
        help="Optional: keep only top K axes by average absolute value across runs",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_run_label(path: Path) -> str:
    parts = path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.stem


def get_metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "delta":
        return float(row["projection_delta_neutral_minus_trait"])
    if metric == "neutral":
        return float(row["projection_score_neutral"])
    if metric == "trait":
        return float(row["projection_score_trait"])
    raise ValueError(f"Unknown metric: {metric}")


def aggregate(values: list[float], mode: str) -> float:
    if not values:
        raise ValueError("Cannot aggregate empty list")
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "median":
        values = sorted(values)
        n = len(values)
        mid = n // 2
        if n % 2 == 1:
            return values[mid]
        return (values[mid - 1] + values[mid]) / 2.0
    raise ValueError(f"Unknown aggregate mode: {mode}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_label",
                "axis_trait",
                "metric",
                "aggregate",
                "n_rows",
                "value",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    summaries: list[dict[str, Any]] = []

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        run_label = infer_run_label(input_path)
        rows = load_jsonl(input_path)

        by_axis: dict[str, list[float]] = {}
        counts: dict[str, int] = {}

        for row in rows:
            axis_trait = row.get("projection_trait")
            if axis_trait is None:
                continue

            value = get_metric_value(row, args.metric)
            by_axis.setdefault(axis_trait, []).append(value)
            counts[axis_trait] = counts.get(axis_trait, 0) + 1

        for axis_trait, values in by_axis.items():
            summaries.append(
                {
                    "run_label": run_label,
                    "axis_trait": axis_trait,
                    "metric": args.metric,
                    "aggregate": args.aggregate,
                    "n_rows": counts[axis_trait],
                    "value": aggregate(values, args.aggregate),
                }
            )

    if not summaries:
        raise ValueError("No projection rows found across inputs.")

    run_labels = sorted({row["run_label"] for row in summaries})
    axis_traits = sorted({row["axis_trait"] for row in summaries})

    if args.top_k_axes is not None:
        axis_strength: dict[str, list[float]] = {}
        for row in summaries:
            axis_strength.setdefault(row["axis_trait"], []).append(abs(float(row["value"])))

        ranked_axes = sorted(
            axis_traits,
            key=lambda axis: sum(axis_strength.get(axis, [])) / max(len(axis_strength.get(axis, [])), 1),
            reverse=True,
        )
        axis_traits = ranked_axes[: args.top_k_axes]

    matrix = []
    summary_lookup = {
        (row["run_label"], row["axis_trait"]): float(row["value"])
        for row in summaries
        if row["axis_trait"] in axis_traits
    }

    for run_label in run_labels:
        matrix_row = []
        for axis_trait in axis_traits:
            matrix_row.append(summary_lookup.get((run_label, axis_trait), 0.0))
        matrix.append(matrix_row)

    plt.figure(figsize=(max(10, len(axis_traits) * 0.5), max(6, len(run_labels) * 0.5)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=f"{args.aggregate} {args.metric}")
    plt.xticks(range(len(axis_traits)), axis_traits, rotation=45, ha="right")
    plt.yticks(range(len(run_labels)), run_labels)
    plt.xlabel("Projection axis")
    plt.ylabel("User trait run")
    plt.title(args.title or "Many traits on many axes")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        write_csv(summaries, Path(args.csv_output))

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved CSV summary to {args.csv_output}")


if __name__ == "__main__":
    main()