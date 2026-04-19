#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot_utils import aggregate, load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot threshold-band progression across projection axes."
    )
    parser.add_argument("--inputs", type=str, nargs="+", required=True, help="Projection JSONL files for each band")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Display labels for each input")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional CSV summary output path")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--top-k-axes", type=int, default=20)
    parser.add_argument("--rank-by", choices=["spread", "abs_mean"], default="spread")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def collect_axis_values(rows: list[dict[str, Any]], aggregate_mode: str) -> dict[str, float]:
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue
        by_axis.setdefault(axis, []).append(float(row["projection_score_trait"]))

    return {axis: aggregate(values, aggregate_mode) for axis, values in by_axis.items()}


def main() -> None:
    args = parse_args()

    if len(args.inputs) != len(args.labels):
        raise ValueError("--inputs and --labels must have the same length")

    series: list[tuple[str, dict[str, float]]] = []
    for input_path, label in zip(args.inputs, args.labels):
        rows = load_jsonl(Path(input_path))
        if not rows:
            raise ValueError(f"No rows found in {input_path}")
        series.append((label, collect_axis_values(rows, args.aggregate)))

    common_axes = None
    for _, axis_map in series:
        axes = set(axis_map.keys())
        common_axes = axes if common_axes is None else common_axes & axes

    if not common_axes:
        raise ValueError("No common projection axes found across all threshold bands")

    axis_scores_for_ranking = {
        axis: [axis_map[axis] for _, axis_map in series]
        for axis in common_axes
    }

    if args.rank_by == "spread":
        ranked_axes = sorted(
            common_axes,
            key=lambda axis: max(axis_scores_for_ranking[axis]) - min(axis_scores_for_ranking[axis]),
            reverse=True,
        )
    else:
        ranked_axes = sorted(
            common_axes,
            key=lambda axis: sum(abs(v) for v in axis_scores_for_ranking[axis]) / len(axis_scores_for_ranking[axis]),
            reverse=True,
        )

    axis_traits = ranked_axes[: args.top_k_axes]
    x_positions = list(range(len(axis_traits)))

    plt.figure(figsize=(max(12, len(axis_traits) * 0.7), 7))

    csv_rows: list[dict[str, Any]] = []
    for label, axis_map in series:
        values = [axis_map[axis] for axis in axis_traits]
        plt.plot(x_positions, values, marker="o", linewidth=1.8, label=label)
        for axis, value in zip(axis_traits, values):
            csv_rows.append({"axis_trait": axis, "series_label": label, "value": value})

    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, axis_traits, rotation=45, ha="right")
    plt.xlabel("Projection axis")
    plt.ylabel(f"{args.aggregate} projection score")
    plt.title(args.title or "Threshold-band progression across projection axes")
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        write_csv(csv_rows, Path(args.csv_output), fieldnames=["axis_trait", "series_label", "value"])

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved CSV summary to {args.csv_output}")


if __name__ == "__main__":
    main()
