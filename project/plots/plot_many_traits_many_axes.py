#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot_utils import aggregate, infer_run_label, load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting a trait-by-axis comparison heatmap."""
    parser = argparse.ArgumentParser(
        description="Plot many user-trait runs across many projection axes, always including Neutral."
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
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation over rows within each run/axis",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=None,
        help="Optional: keep only top K axes by average absolute trait score across runs",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()
def main() -> None:
    """Aggregate many-axis projections across runs and render a heatmap."""
    args = parse_args()

    summaries: list[dict[str, Any]] = []
    neutral_by_axis_all_runs: dict[str, list[float]] = {}

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        run_label = infer_run_label(input_path)
        rows = load_jsonl(input_path)

        by_axis_trait: dict[str, list[float]] = {}
        counts_trait: dict[str, int] = {}

        for row in rows:
            axis_trait = row.get("projection_trait")
            if axis_trait is None:
                continue

            trait_score = float(row["projection_score_trait"])
            neutral_score = float(row["projection_score_neutral"])

            by_axis_trait.setdefault(axis_trait, []).append(trait_score)
            counts_trait[axis_trait] = counts_trait.get(axis_trait, 0) + 1

            neutral_by_axis_all_runs.setdefault(axis_trait, []).append(neutral_score)

        for axis_trait, values in by_axis_trait.items():
            summaries.append(
                {
                    "run_label": run_label,
                    "axis_trait": axis_trait,
                    "kind": "trait",
                    "aggregate": args.aggregate,
                    "n_rows": counts_trait[axis_trait],
                    "value": aggregate(values, args.aggregate),
                }
            )

    if not summaries:
        raise ValueError("No projection rows found across inputs.")

    for axis_trait, values in neutral_by_axis_all_runs.items():
        summaries.append(
            {
                "run_label": "Neutral",
                "axis_trait": axis_trait,
                "kind": "neutral",
                "aggregate": args.aggregate,
                "n_rows": len(values),
                "value": aggregate(values, args.aggregate),
            }
        )

    run_labels = sorted({row["run_label"] for row in summaries if row["run_label"] != "Neutral"})
    run_labels = ["Neutral"] + run_labels
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

    summary_lookup = {
        (row["run_label"], row["axis_trait"]): float(row["value"])
        for row in summaries
        if row["axis_trait"] in axis_traits
    }

    matrix = []
    for run_label in run_labels:
        matrix_row = []
        for axis_trait in axis_traits:
            matrix_row.append(summary_lookup.get((run_label, axis_trait), 0.0))
        matrix.append(matrix_row)

    plt.figure(figsize=(max(10, len(axis_traits) * 0.5), max(6, len(run_labels) * 0.5)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label=f"{args.aggregate} projection score")
    plt.xticks(range(len(axis_traits)), axis_traits, rotation=45, ha="right")
    plt.yticks(range(len(run_labels)), run_labels)
    plt.xlabel("Projection axis")
    plt.ylabel("Series")
    plt.title(args.title or "Many traits on many axes (with Neutral)")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        write_csv(
            summaries,
            Path(args.csv_output),
            fieldnames=["run_label", "axis_trait", "kind", "aggregate", "n_rows", "value"],
        )

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved CSV summary to {args.csv_output}")


if __name__ == "__main__":
    main()
