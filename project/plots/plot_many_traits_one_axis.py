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
        description="Plot many user-trait runs on one fixed projection axis."
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
        "--metric",
        choices=["delta", "neutral", "trait"],
        default="delta",
        help="Which value to aggregate per run",
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_run_label(path: Path) -> str:
    """
    Assumes file path like:
    outputs/user_prompts/<trait>/projections/<run_name>.jsonl
    Returns the trait folder name when possible, otherwise file stem.
    """
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

        filtered = [r for r in rows if r.get("projection_trait") == args.axis_trait]
        if not filtered:
            print(f"Skipping {input_path}: no rows for axis {args.axis_trait}")
            continue

        values = [get_metric_value(r, args.metric) for r in filtered]
        value = aggregate(values, args.aggregate)

        summaries.append(
            {
                "run_label": run_label,
                "axis_trait": args.axis_trait,
                "metric": args.metric,
                "aggregate": args.aggregate,
                "n_rows": len(filtered),
                "value": value,
            }
        )

    if not summaries:
        raise ValueError("No matching rows found across inputs.")

    summaries.sort(key=lambda x: x["value"], reverse=True)

    labels = [row["run_label"] for row in summaries]
    values = [row["value"] for row in summaries]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{args.aggregate} {args.metric}")
    plt.xlabel("User trait run")
    plt.title(args.title or f"Many traits on one axis: {args.axis_trait}")
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