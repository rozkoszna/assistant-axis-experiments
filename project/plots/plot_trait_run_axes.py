#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot_utils import aggregate, infer_run_label, load_jsonl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting one run across its strongest axes."""
    parser = argparse.ArgumentParser(
        description="Plot aggregated projection results for one user-trait run across axes."
    )
    parser.add_argument("--input", type=str, required=True, help="Input projection JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--top-k", type=int, default=20, help="Number of axes to plot")
    parser.add_argument(
        "--metric",
        choices=["delta", "neutral", "trait"],
        default="delta",
        help="Which metric to aggregate",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate rows within each axis",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title",
    )
    return parser.parse_args()
def get_metric_value(row: dict[str, Any], metric: str) -> float:
    """Read the requested scalar metric from one projection row."""
    if metric == "delta":
        return float(row["projection_delta_trait_minus_neutral"])
    if metric == "neutral":
        return float(row["projection_score_neutral"])
    if metric == "trait":
        return float(row["projection_score_trait"])
    raise ValueError(f"Unknown metric: {metric}")
def infer_trait_run_name(path: Path, rows: list[dict[str, Any]]) -> str:
    """Infer a display name for the plotted run from row metadata or path."""
    if rows and "trait" in rows[0]:
        return str(rows[0]["trait"])
    return infer_run_label(path)


def main() -> None:
    """Aggregate one run's axis scores and render a horizontal bar chart."""
    args = parse_args()

    input_path = Path(args.input)
    rows = load_jsonl(input_path)

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    by_axis: dict[str, list[float]] = {}

    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue

        value = get_metric_value(row, args.metric)
        by_axis.setdefault(axis, []).append(value)

    if not by_axis:
        raise ValueError("No projection_trait rows found in input")

    summary_rows = []
    for axis, values in by_axis.items():
        summary_rows.append(
            {
                "axis": axis,
                "value": aggregate(values, args.aggregate),
                "n": len(values),
            }
        )

    summary_rows.sort(key=lambda x: abs(x["value"]), reverse=True)
    summary_rows = summary_rows[: args.top_k]

    axes = [row["axis"] for row in summary_rows]
    values = [row["value"] for row in summary_rows]

    run_name = infer_trait_run_name(input_path, rows)

    plt.figure(figsize=(10, max(5, 0.4 * len(summary_rows))))
    plt.barh(axes, values)
    plt.axvline(0)

    plt.xlabel(f"{args.aggregate} {args.metric}")
    plt.ylabel("Projection axis")
    plt.title(args.title or f"{run_name}: top {args.top_k} axes by {args.aggregate} {args.metric}")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
