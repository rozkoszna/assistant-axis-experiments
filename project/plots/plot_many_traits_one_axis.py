#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot_utils import aggregate, infer_run_label, load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting multiple traits on one fixed axis."""
    parser = argparse.ArgumentParser(
        description="Plot many user-trait runs on one fixed projection axis, always including Neutral."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Projection JSONL files from multiple trait runs",
    )
    parser.add_argument(
        "--axis-trait",
        type=str,
        required=True,
        help="Projection axis to keep, e.g. supportive",
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
        help="Aggregation over rows within each run",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()
def main() -> None:
    """Aggregate one-axis projections across runs and render a simple bar chart."""
    args = parse_args()

    summaries: list[dict[str, Any]] = []
    neutral_values_all_runs: list[float] = []

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        run_label = infer_run_label(input_path)
        rows = load_jsonl(input_path)

        filtered = [r for r in rows if r.get("projection_trait") == args.axis_trait]
        if not filtered:
            print(f"Skipping {input_path}: no rows for axis {args.axis_trait}")
            continue

        trait_values = [float(r["projection_score_trait"]) for r in filtered]
        neutral_values = [float(r["projection_score_neutral"]) for r in filtered]

        summaries.append(
            {
                "run_label": run_label,
                "axis_trait": args.axis_trait,
                "kind": "trait",
                "aggregate": args.aggregate,
                "n_rows": len(filtered),
                "value": aggregate(trait_values, args.aggregate),
            }
        )

        neutral_values_all_runs.extend(neutral_values)

    if not summaries:
        raise ValueError("No matching rows found across inputs.")

    summaries.insert(
        0,
        {
            "run_label": "Neutral",
            "axis_trait": args.axis_trait,
            "kind": "neutral",
            "aggregate": args.aggregate,
            "n_rows": len(neutral_values_all_runs),
            "value": aggregate(neutral_values_all_runs, args.aggregate),
        },
    )

    labels = [row["run_label"] for row in summaries]
    values = [row["value"] for row in summaries]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{args.aggregate} projection score")
    plt.xlabel("Series")
    plt.title(args.title or f"Many traits on one axis: {args.axis_trait} (with Neutral)")
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
