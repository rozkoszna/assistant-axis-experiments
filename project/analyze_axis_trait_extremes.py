#!/usr/bin/env python3
"""
Find user traits that move each projection axis most positively/negatively.

Input:
- one or more projection JSONL files (typically from run_multi_trait_analysis)

For each axis, this script computes per-trait aggregate shift using:
- projection_delta_trait_minus_neutral

Then it reports:
- top K positive traits (largest mean shift)
- top K negative traits (smallest mean shift)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT / "project"
for search_path in (REPO_ROOT, PROJECT_ROOT):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

from plots.plot_utils import load_jsonl, write_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI args for per-axis trait-extremes analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "For each axis, find user traits with the largest positive/negative "
            "mean projection delta (trait minus neutral)."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Projection JSONL files, e.g. outputs/user_prompts/*/projections/<run>.jsonl",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="How many positive/negative traits to report per axis",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum row count required for a trait-axis pair",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional flat CSV output path",
    )
    return parser.parse_args()


def infer_trait(row: dict[str, Any], input_path: Path) -> str | None:
    """Infer user trait label from row first, then fallback to path structure."""
    row_trait = row.get("trait")
    if isinstance(row_trait, str) and row_trait.strip():
        return row_trait.strip()

    parts = input_path.parts
    if "user_prompts" in parts:
        index = parts.index("user_prompts")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def mean(values: list[float]) -> float:
    """Compute arithmetic mean for non-empty values."""
    return sum(values) / len(values)


def main() -> None:
    """Load projection rows, compute per-axis trait extremes, and write outputs."""
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1")

    by_axis_trait: dict[str, dict[str, list[float]]] = {}
    total_rows = 0

    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        rows = load_jsonl(input_path)
        for row in rows:
            axis = row.get("projection_trait")
            if axis is None:
                continue
            trait = infer_trait(row, input_path)
            if trait is None:
                continue
            delta = row.get("projection_delta_trait_minus_neutral")
            if delta is None:
                continue

            axis_key = str(axis)
            trait_key = str(trait)
            by_axis_trait.setdefault(axis_key, {}).setdefault(trait_key, []).append(float(delta))
            total_rows += 1

    if not by_axis_trait:
        raise ValueError("No projection rows with axis/trait/delta found in provided inputs")

    result: dict[str, Any] = {
        "top_k": args.top_k,
        "min_count": args.min_count,
        "num_inputs": len(args.inputs),
        "num_rows_used": total_rows,
        "axes": {},
    }
    flat_rows: list[dict[str, Any]] = []

    for axis in sorted(by_axis_trait.keys()):
        trait_scores: list[dict[str, Any]] = []
        for trait, values in by_axis_trait[axis].items():
            if len(values) < args.min_count:
                continue
            trait_scores.append(
                {
                    "trait": trait,
                    "count": len(values),
                    "mean_delta": mean(values),
                    "min_delta": min(values),
                    "max_delta": max(values),
                }
            )

        if not trait_scores:
            continue

        positive = sorted(
            trait_scores,
            key=lambda item: (item["mean_delta"], item["count"]),
            reverse=True,
        )[: args.top_k]
        negative = sorted(
            trait_scores,
            key=lambda item: (item["mean_delta"], -item["count"]),
        )[: args.top_k]

        result["axes"][axis] = {
            "positive": positive,
            "negative": negative,
            "num_traits_considered": len(trait_scores),
        }

        for rank, item in enumerate(positive, start=1):
            flat_rows.append(
                {
                    "axis": axis,
                    "direction": "positive",
                    "rank": rank,
                    "trait": item["trait"],
                    "count": item["count"],
                    "mean_delta": item["mean_delta"],
                    "min_delta": item["min_delta"],
                    "max_delta": item["max_delta"],
                }
            )
        for rank, item in enumerate(negative, start=1):
            flat_rows.append(
                {
                    "axis": axis,
                    "direction": "negative",
                    "rank": rank,
                    "trait": item["trait"],
                    "count": item["count"],
                    "mean_delta": item["mean_delta"],
                    "min_delta": item["min_delta"],
                    "max_delta": item["max_delta"],
                }
            )

    print("\nPer-axis trait extremes")
    print("=======================")
    for axis in sorted(result["axes"].keys()):
        axis_result = result["axes"][axis]
        print(f"\nAxis: {axis}")
        print("  Positive:")
        for item in axis_result["positive"]:
            print(
                f"    - {item['trait']}: mean_delta={item['mean_delta']:.4f} "
                f"(n={item['count']})"
            )
        print("  Negative:")
        for item in axis_result["negative"]:
            print(
                f"    - {item['trait']}: mean_delta={item['mean_delta']:.4f} "
                f"(n={item['count']})"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON to {output_path}")

    if args.output_csv:
        write_csv(
            flat_rows,
            Path(args.output_csv),
            fieldnames=[
                "axis",
                "direction",
                "rank",
                "trait",
                "count",
                "mean_delta",
                "min_delta",
                "max_delta",
            ],
        )
        print(f"Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
