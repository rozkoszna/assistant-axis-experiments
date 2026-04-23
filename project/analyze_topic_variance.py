#!/usr/bin/env python3
"""
Decompose projection variance into between-topic and within-topic components.

This script works from precomputed projection JSONL files. For each condition
and projection axis, it groups rows by a topic field such as `intent_index` and
reports:
- overall mean/std across examples
- between-topic mean/std across topic means
- within-topic std averaged across topics
- pooled within-topic std across all examples after removing topic means

Use this when you want to know whether variation comes mostly from topic choice
or from examples inside the same topic.
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
    """Parse CLI arguments for topic-variance analysis."""
    parser = argparse.ArgumentParser(
        description="Compute between-topic and within-topic variance from projection JSONL files."
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
        help="Also analyse projection_score_neutral from the same files.",
    )
    parser.add_argument("--neutral-label", type=str, default="Neutral")
    parser.add_argument(
        "--group-by-field",
        type=str,
        default="intent_index",
        help="Field used as the topic/group id. Default: intent_index.",
    )
    parser.add_argument(
        "--axis-filter",
        nargs="*",
        default=None,
        help="Optional list of axis traits to keep.",
    )
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--per-topic-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    """Return arithmetic mean."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    """Return sample variance, or 0 for a single value."""
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return sum((value - m) ** 2 for value in values) / (len(values) - 1)


def sample_std(values: list[float]) -> float:
    """Return sample standard deviation, or 0 for a single value."""
    return math.sqrt(sample_variance(values))


def collect_grouped_values(
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    group_by_field: str,
) -> dict[str, dict[str, list[float]]]:
    """Collect projection values as axis -> topic -> values."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        group = row.get(group_by_field)
        if axis is None or group is None:
            continue
        grouped.setdefault(str(axis), {}).setdefault(str(group), []).append(float(row[value_key]))
    return grouped


def summarize_axis_topics(
    condition: str,
    axis: str,
    topics: dict[str, list[float]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one summary row and per-topic rows for one condition/axis."""
    topic_rows: list[dict[str, Any]] = []
    all_values: list[float] = []
    topic_means: list[float] = []
    topic_stds: list[float] = []
    residuals: list[float] = []

    for topic, values in sorted(topics.items()):
        topic_mean = mean(values)
        topic_std = sample_std(values)
        topic_var = sample_variance(values)

        all_values.extend(values)
        topic_means.append(topic_mean)
        topic_stds.append(topic_std)
        residuals.extend(value - topic_mean for value in values)

        topic_rows.append(
            {
                "condition": condition,
                "axis": axis,
                "topic": topic,
                "count": len(values),
                "mean": topic_mean,
                "std": topic_std,
                "variance": topic_var,
                "min": min(values),
                "max": max(values),
            }
        )

    between_var = sample_variance(topic_means)
    within_vars = [sample_variance(values) for values in topics.values()]
    summary = {
        "condition": condition,
        "axis": axis,
        "num_topics": len(topics),
        "count": len(all_values),
        "overall_mean": mean(all_values),
        "overall_std": sample_std(all_values),
        "overall_variance": sample_variance(all_values),
        "between_topic_std": math.sqrt(between_var),
        "between_topic_variance": between_var,
        "mean_within_topic_std": mean(topic_stds) if topic_stds else 0.0,
        "mean_within_topic_variance": mean(within_vars) if within_vars else 0.0,
        "pooled_within_topic_std": sample_std(residuals),
        "pooled_within_topic_variance": sample_variance(residuals),
    }
    return summary, topic_rows


def main() -> None:
    """Compute topic variance decomposition and write optional outputs."""
    args = parse_args()

    if len(args.trait_inputs) != len(args.trait_labels):
        raise ValueError("--trait-inputs and --trait-labels must have the same length")

    condition_axis_topics: list[tuple[str, dict[str, dict[str, list[float]]]]] = []

    if args.include_neutral:
        merged_neutral: dict[str, dict[str, list[float]]] = {}
    else:
        merged_neutral = {}

    for input_path, label in zip(args.trait_inputs, args.trait_labels):
        rows = load_jsonl(Path(input_path))
        if not rows:
            raise ValueError(f"No rows found in input: {input_path}")

        trait_grouped = collect_grouped_values(
            rows,
            "projection_score_trait",
            group_by_field=args.group_by_field,
        )
        condition_axis_topics.append((label, trait_grouped))

        if args.include_neutral:
            neutral_grouped = collect_grouped_values(
                rows,
                "projection_score_neutral",
                group_by_field=args.group_by_field,
            )
            for axis, topic_map in neutral_grouped.items():
                for topic, values in topic_map.items():
                    merged_neutral.setdefault(axis, {}).setdefault(topic, []).extend(values)

    if args.include_neutral:
        condition_axis_topics.insert(0, (args.neutral_label, merged_neutral))

    if args.axis_filter:
        axes = list(dict.fromkeys(args.axis_filter))
    else:
        axis_sets = [set(axis_topics.keys()) for _, axis_topics in condition_axis_topics]
        axes = sorted(set.intersection(*axis_sets)) if axis_sets else []

    if not axes:
        raise ValueError("No axes found to analyse")

    summary_rows: list[dict[str, Any]] = []
    per_topic_rows: list[dict[str, Any]] = []

    for condition, axis_topics in condition_axis_topics:
        for axis in axes:
            if axis not in axis_topics:
                continue
            summary, topic_rows = summarize_axis_topics(condition, axis, axis_topics[axis])
            summary_rows.append(summary)
            per_topic_rows.extend(topic_rows)

    print("\nTopic variance summary")
    print("======================")
    for row in summary_rows:
        print(
            f"{row['condition']:>15} | {row['axis']:<16} "
            f"n={row['count']:<3} topics={row['num_topics']:<2} "
            f"mean={row['overall_mean']:.4f} overall_std={row['overall_std']:.4f} "
            f"between_topic_std={row['between_topic_std']:.4f} "
            f"pooled_within_topic_std={row['pooled_within_topic_std']:.4f}"
        )

    if args.output_csv:
        write_csv(
            summary_rows,
            Path(args.output_csv),
            fieldnames=[
                "condition",
                "axis",
                "num_topics",
                "count",
                "overall_mean",
                "overall_std",
                "overall_variance",
                "between_topic_std",
                "between_topic_variance",
                "mean_within_topic_std",
                "mean_within_topic_variance",
                "pooled_within_topic_std",
                "pooled_within_topic_variance",
            ],
        )
        print(f"\nSaved summary CSV to {args.output_csv}")

    if args.per_topic_csv:
        write_csv(
            per_topic_rows,
            Path(args.per_topic_csv),
            fieldnames=["condition", "axis", "topic", "count", "mean", "std", "variance", "min", "max"],
        )
        print(f"Saved per-topic CSV to {args.per_topic_csv}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"summary": summary_rows, "per_topic": per_topic_rows}, f, indent=2)
        print(f"Saved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
