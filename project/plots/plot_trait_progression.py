#!/usr/bin/env python3
"""
Plot one or more projection series across a shared set of assistant axes.

This is the main comparison plotter for the user-trait pipeline.

It supports both common use cases:

1. Trait progression:
   Compare a baseline series plus several trait or intensity runs, for example
   `neutral` vs `little confused` vs `confused` vs `very confused`.

2. Score-range comparison:
   Compare several selected groups from the same judged pool, for example
   `89-92`, `92-94`, and `94-100`, optionally with an aggregated Neutral line.

Input model
-----------
- `--baseline-input` reads one JSONL file and plots its
  `projection_score_neutral` values as a baseline series.
- `--trait-inputs` reads one or more JSONL files and plots one chosen value
  from each file:
  - `projection_score_trait`
  - `projection_score_neutral`
  - `projection_delta_trait_minus_neutral`
- `--include-neutral` aggregates `projection_score_neutral` across all
  `--trait-inputs` and inserts that as an additional series.

Ranking
-------
Axes are restricted to the axes shared by every series, then ranked by:
- `spread`: max minus min across plotted series
- `abs_mean`: average absolute value across plotted series

Notes
-----
- If you want one plotting entrypoint, use this script directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from plot_utils import aggregate, load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for trait, neutral, and score-range comparison plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot several projection series across many axes. "
            "Use this for both trait-progression and score-range comparison plots."
        )
    )
    parser.add_argument(
        "--baseline-input",
        type=str,
        default=None,
        help=(
            "Optional projection JSONL file used to read a baseline series from "
            "`projection_score_neutral`."
        ),
    )
    parser.add_argument(
        "--trait-inputs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Projection JSONL files for the non-baseline series. These may be "
            "trait runs, intensity runs, or score-band runs."
        ),
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=None,
        help="Compatibility alias for --trait-inputs.",
    )
    parser.add_argument(
        "--trait-labels",
        type=str,
        nargs="+",
        default=None,
        help="Display labels for the plotted non-baseline series, same order as --trait-inputs",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Compatibility alias for --trait-labels.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="neutral",
        help="Display label for the baseline series",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help=(
            "Also include a Neutral series aggregated from "
            "`projection_score_neutral` across all `--trait-inputs`."
        ),
    )
    parser.add_argument(
        "--trait-value-key",
        choices=["projection_score_trait", "projection_score_neutral", "projection_delta_trait_minus_neutral"],
        default="projection_score_trait",
        help=(
            "Which value to read from each non-baseline input file. "
            "`projection_score_trait` is the usual choice for trait runs and "
            "score-band runs."
        ),
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
        help="Aggregation over rows within each axis",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=20,
        help="Keep top K shared axes ranked across the plotted series",
    )
    parser.add_argument(
        "--rank-by",
        choices=["spread", "abs_mean"],
        default="spread",
        help="How to rank axes: spread=max-min across series, or mean absolute value",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title",
    )
    return parser.parse_args()
def collect_axis_values(
    rows: list[dict[str, Any]],
    value_key: str,
    aggregate_mode: str,
) -> dict[str, float]:
    """Aggregate one projection file into axis -> score values for plotting."""
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        axis = row.get("projection_trait")
        if axis is None:
            continue
        by_axis.setdefault(axis, []).append(float(row[value_key]))

    return {
        axis: aggregate(values, aggregate_mode)
        for axis, values in by_axis.items()
    }
def main() -> None:
    """Render a multi-series comparison plot across shared projection axes."""
    args = parse_args()

    trait_inputs = args.trait_inputs or args.inputs
    trait_labels = args.trait_labels or args.labels

    if not trait_inputs or not trait_labels:
        raise ValueError(
            "Provide series with --trait-inputs/--trait-labels "
            "(or the compatibility aliases --inputs/--labels)."
        )

    if len(trait_inputs) != len(trait_labels):
        raise ValueError("--trait-inputs and --trait-labels must have the same length")

    series: list[tuple[str, dict[str, float]]] = []
    neutral_rows_all_inputs: list[dict[str, Any]] = []

    if args.baseline_input is not None:
        baseline_rows = load_jsonl(Path(args.baseline_input))
        if not baseline_rows:
            raise ValueError(f"No rows found in baseline input: {args.baseline_input}")

        baseline_by_axis = collect_axis_values(
            baseline_rows,
            value_key="projection_score_neutral",
            aggregate_mode=args.aggregate,
        )
        series.append((args.baseline_label, baseline_by_axis))

    for trait_input, trait_label in zip(trait_inputs, trait_labels):
        rows = load_jsonl(Path(trait_input))
        if not rows:
            raise ValueError(f"No rows found in trait input: {trait_input}")

        trait_by_axis = collect_axis_values(
            rows,
            value_key=args.trait_value_key,
            aggregate_mode=args.aggregate,
        )
        series.append((trait_label, trait_by_axis))
        neutral_rows_all_inputs.extend(rows)

    if args.include_neutral:
        series.insert(
            0,
            (
                args.baseline_label,
                collect_axis_values(
                    neutral_rows_all_inputs,
                    value_key="projection_score_neutral",
                    aggregate_mode=args.aggregate,
                ),
            ),
        )

    if not series:
        raise ValueError("No series were constructed. Provide trait inputs and/or a baseline input.")

    common_axes = None
    for _, axis_map in series:
        axes = set(axis_map.keys())
        common_axes = axes if common_axes is None else common_axes & axes

    if not common_axes:
        raise ValueError("No common projection axes found across all series")

    axis_scores_for_ranking: dict[str, list[float]] = {}
    for axis in common_axes:
        axis_scores_for_ranking[axis] = [axis_map[axis] for _, axis_map in series]

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

    series_values: list[tuple[str, list[float]]] = []
    for label, axis_map in series:
        values = [axis_map[axis] for axis in axis_traits]
        series_values.append((label, values))
        plt.plot(x_positions, values, marker="o", linewidth=1.8, label=label)

    for i, axis in enumerate(axis_traits):
        ys = [values[i] for _, values in series_values]
        ys_sorted = sorted(ys)
        plt.vlines(i, ys_sorted[0], ys_sorted[-1], alpha=0.35, linewidth=1)

    plt.axhline(0, linewidth=1)
    plt.xticks(x_positions, axis_traits, rotation=45, ha="right")
    plt.xlabel("Projection axis")
    plt.ylabel(f"{args.aggregate} projection score")
    plt.title(args.title or "Trait progression across projection axes")
    plt.legend()
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    if args.csv_output:
        csv_rows: list[dict[str, Any]] = []
        for label, values in series_values:
            for axis, value in zip(axis_traits, values):
                csv_rows.append(
                    {
                        "axis_trait": axis,
                        "series_label": label,
                        "value": value,
                    }
                )
        write_csv(csv_rows, Path(args.csv_output), fieldnames=["axis_trait", "series_label", "value"])

    print(f"Saved plot to {output_path}")
    if args.csv_output:
        print(f"Saved CSV summary to {args.csv_output}")


if __name__ == "__main__":
    main()
