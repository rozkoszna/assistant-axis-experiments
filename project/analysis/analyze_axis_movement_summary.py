#!/usr/bin/env python3
"""
Summarize which projection axes are moved the most by user traits.

For each axis, this script aggregates per-trait mean deltas
(`projection_delta_trait_minus_neutral`) and reports movement metrics such as:
- mean absolute shift across traits
- max absolute shift
- signed mean shift
- spread of trait means
"""

from __future__ import annotations

import argparse
import json
import math
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
    """Parse CLI arguments for axis movement summary analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank axes by how strongly user traits move them, based on "
            "projection_delta_trait_minus_neutral."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Projection JSONL files")
    parser.add_argument("--top-k", type=int, default=25, help="How many top axes to print")
    parser.add_argument("--min-traits", type=int, default=2, help="Minimum trait count per axis")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV output path")
    return parser.parse_args()


def infer_trait(row: dict[str, Any], input_path: Path) -> str | None:
    """Infer trait from row field first, then from projection file path."""
    row_trait = row.get("trait")
    if isinstance(row_trait, str) and row_trait.strip():
        return row_trait.strip()
    parts = input_path.parts
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    var = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(var)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_traits < 1:
        raise ValueError("--min-traits must be >= 1")

    by_axis_trait_values: dict[str, dict[str, list[float]]] = {}
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        rows = load_jsonl(input_path)
        for row in rows:
            axis = row.get("projection_trait")
            delta = row.get("projection_delta_trait_minus_neutral")
            if axis is None or delta is None:
                continue
            trait = infer_trait(row, input_path)
            if trait is None:
                continue
            by_axis_trait_values.setdefault(str(axis), {}).setdefault(trait, []).append(float(delta))

    summary_rows: list[dict[str, Any]] = []
    for axis, trait_map in by_axis_trait_values.items():
        trait_means: dict[str, float] = {}
        for trait, values in trait_map.items():
            if not values:
                continue
            trait_means[trait] = mean(values)

        if len(trait_means) < args.min_traits:
            continue

        trait_mean_values = list(trait_means.values())
        mean_abs_shift = mean([abs(value) for value in trait_mean_values])
        max_abs_shift = max(abs(value) for value in trait_mean_values)
        signed_mean_shift = mean(trait_mean_values)
        trait_mean_std = std(trait_mean_values)

        strongest_positive_trait = max(trait_means.items(), key=lambda item: item[1])
        strongest_negative_trait = min(trait_means.items(), key=lambda item: item[1])

        summary_rows.append(
            {
                "axis": axis,
                "num_traits": len(trait_means),
                "mean_abs_shift": mean_abs_shift,
                "max_abs_shift": max_abs_shift,
                "signed_mean_shift": signed_mean_shift,
                "trait_mean_std": trait_mean_std,
                "strongest_positive_trait": strongest_positive_trait[0],
                "strongest_positive_mean_delta": strongest_positive_trait[1],
                "strongest_negative_trait": strongest_negative_trait[0],
                "strongest_negative_mean_delta": strongest_negative_trait[1],
            }
        )

    summary_rows = sorted(summary_rows, key=lambda row: row["mean_abs_shift"], reverse=True)

    print("\nAxes moved most by traits (ranked by mean_abs_shift)")
    print("====================================================")
    for rank, row in enumerate(summary_rows[: args.top_k], start=1):
        print(
            f"{rank:>2}. {row['axis']:<20} mean_abs={row['mean_abs_shift']:.4f} "
            f"max_abs={row['max_abs_shift']:.4f} n_traits={row['num_traits']}"
        )

    if args.output_csv:
        write_csv(
            summary_rows,
            Path(args.output_csv),
            fieldnames=[
                "axis",
                "num_traits",
                "mean_abs_shift",
                "max_abs_shift",
                "signed_mean_shift",
                "trait_mean_std",
                "strongest_positive_trait",
                "strongest_positive_mean_delta",
                "strongest_negative_trait",
                "strongest_negative_mean_delta",
            ],
        )
        print(f"\nSaved CSV to {args.output_csv}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        print(f"Saved JSON to {output_path}")


if __name__ == "__main__":
    main()
