#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from plot_utils import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-run variation from measure_trait_axis_variation_from_pipeline outputs."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to per_run_summary.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output PNG",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title",
    )
    return parser.parse_args()
def main() -> None:
    args = parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    run_names = [row["run_name"] for row in rows]
    means = [row["mean"] for row in rows]
    mins = [row["min"] for row in rows]
    maxs = [row["max"] for row in rows]

    x = list(range(len(rows)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, means, marker="o", label="mean score")
    plt.vlines(x, mins, maxs, alpha=0.6, label="min–max range")

    plt.axhline(0.0, linewidth=1)

    plt.xticks(x, run_names, rotation=45, ha="right")
    plt.xlabel("Pipeline run")
    plt.ylabel("Projection score")

    if args.title is not None:
        plt.title(args.title)
    else:
        trait = rows[0].get("trait", "trait")
        axis_trait = rows[0].get("axis_trait", "axis")
        plt.title(f"Variation across runs: {trait} on {axis_trait} axis")

    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
