#!/usr/bin/env python3
"""
Plot topic-wise shifts between a trait condition and a neutral baseline.

Input is the per-topic CSV produced by:
    project/analyze_topic_variance.py --per-topic-csv ...

For each topic and axis, this script computes:
    shift = trait_mean - neutral_mean

and renders grouped bars by topic.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for topic-shift plotting."""
    parser = argparse.ArgumentParser(description="Plot per-topic trait-minus-neutral projection shifts.")
    parser.add_argument("--per-topic-csv", type=str, required=True)
    parser.add_argument("--neutral-label", type=str, default="Neutral")
    parser.add_argument("--trait-label", type=str, required=True)
    parser.add_argument("--axes", nargs="+", required=True, help="Axes to plot, e.g. disorganized chaotic")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--csv-output", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load per-topic summary CSV rows."""
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write computed topic shifts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["topic", "axis", "neutral_mean", "trait_mean", "shift_trait_minus_neutral"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_topic_key(topic: str) -> tuple[int, str]:
    """Sort numeric topics numerically and fallback topics lexicographically."""
    try:
        return (0, f"{int(topic):08d}")
    except ValueError:
        return (1, topic)


def main() -> None:
    """Compute and plot topic-wise shifts from a per-topic variance CSV."""
    args = parse_args()
    rows = load_rows(Path(args.per_topic_csv))
    if not rows:
        raise ValueError(f"No rows found in {args.per_topic_csv}")

    means: dict[tuple[str, str, str], float] = {}
    topics: set[str] = set()
    axes = list(dict.fromkeys(args.axes))

    for row in rows:
        condition = row["condition"]
        axis = row["axis"]
        topic = row["topic"]
        if axis not in axes:
            continue
        if condition not in {args.neutral_label, args.trait_label}:
            continue
        means[(condition, axis, topic)] = float(row["mean"])
        topics.add(topic)

    topic_order = sorted(topics, key=sort_topic_key)
    if not topic_order:
        raise ValueError("No matching topics found for requested labels and axes")

    shift_rows: list[dict[str, Any]] = []
    for topic in topic_order:
        for axis in axes:
            neutral_key = (args.neutral_label, axis, topic)
            trait_key = (args.trait_label, axis, topic)
            if neutral_key not in means or trait_key not in means:
                continue
            neutral_mean = means[neutral_key]
            trait_mean = means[trait_key]
            shift_rows.append(
                {
                    "topic": topic,
                    "axis": axis,
                    "neutral_mean": neutral_mean,
                    "trait_mean": trait_mean,
                    "shift_trait_minus_neutral": trait_mean - neutral_mean,
                }
            )

    if not shift_rows:
        raise ValueError("No matching neutral/trait topic pairs found")

    shift_lookup = {
        (row["topic"], row["axis"]): float(row["shift_trait_minus_neutral"])
        for row in shift_rows
    }

    x = np.arange(len(topic_order))
    num_axes = len(axes)
    group_width = 0.8
    bar_width = group_width / max(num_axes, 1)
    offsets = (np.arange(num_axes) - (num_axes - 1) / 2.0) * bar_width

    plt.figure(figsize=(max(9, len(topic_order) * 1.2), 5.5))
    cmap = plt.get_cmap("tab10")

    for idx, axis in enumerate(axes):
        values = [shift_lookup.get((topic, axis), 0.0) for topic in topic_order]
        plt.bar(
            x + offsets[idx],
            values,
            width=bar_width * 0.9,
            label=axis,
            color=cmap(idx % 10),
            alpha=0.9,
        )

    plt.axhline(0.0, color="#555555", linewidth=1.0)
    plt.xticks(x, [f"Topic {topic}" for topic in topic_order], rotation=25, ha="right")
    plt.ylabel(f"{args.trait_label} - {args.neutral_label} projection shift")
    plt.xlabel("Topic")
    plt.title(args.title or f"Topic-wise shifts: {args.trait_label} vs {args.neutral_label}")
    plt.legend(title="Axis")
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        write_rows(Path(args.csv_output), shift_rows)

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved topic-shift CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
