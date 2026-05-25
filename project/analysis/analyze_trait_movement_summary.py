#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize user traits that move axes the most globally, producing 9 tables: "
            "direction(any/positive/negative) x metric(top5_count/top10_count/abs_mean_sum)."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Projection JSONL files (typically one per user trait run).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV/JSON summaries will be saved.",
    )
    parser.add_argument(
        "--prefix",
        default="trait_global_movers",
        help="Filename prefix for generated outputs.",
    )
    return parser.parse_args()


def infer_trait_label(path: str) -> str:
    parts = Path(path).parts
    # .../outputs/user_prompts/<trait>/projections/<file>.jsonl
    if "user_prompts" in parts:
        idx = parts.index("user_prompts")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    stem = Path(path).stem
    return stem.split("__")[-1]


def load_axis_trait_means(path: str) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            axis = str(row["projection_trait"])
            delta = float(row["projection_delta_trait_minus_neutral"])
            sums[axis] += delta
            counts[axis] += 1
    return {axis: sums[axis] / counts[axis] for axis in sums}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_ranked_rows(score_by_trait: dict[str, float], score_name: str) -> list[dict]:
    ordered = sorted(score_by_trait.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for rank, (trait, score) in enumerate(ordered, 1):
        rows.append({"rank": rank, "trait": trait, score_name: score})
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # trait -> axis -> mean delta
    trait_axis_means: dict[str, dict[str, float]] = {}
    for path in args.inputs:
        trait = infer_trait_label(path)
        if path.endswith("__neutral.jsonl"):
            continue
        trait_axis_means[trait] = load_axis_trait_means(path)

    if not trait_axis_means:
        raise ValueError("No trait projection files loaded.")

    common_axes = set.intersection(*(set(d.keys()) for d in trait_axis_means.values()))
    axes = sorted(common_axes)
    if not axes:
        raise ValueError("No shared axes across trait files.")

    traits = sorted(trait_axis_means.keys())

    top5_any = defaultdict(int)
    top10_any = defaultdict(int)
    top5_pos = defaultdict(int)
    top10_pos = defaultdict(int)
    top5_neg = defaultdict(int)
    top10_neg = defaultdict(int)

    abs_mean_sum_any = defaultdict(float)
    abs_mean_sum_pos = defaultdict(float)
    abs_mean_sum_neg = defaultdict(float)

    for axis in axes:
        vals = [(trait, trait_axis_means[trait][axis]) for trait in traits]

        # any direction (by absolute value)
        vals_any = sorted(vals, key=lambda x: abs(x[1]), reverse=True)
        for trait, value in vals_any:
            abs_mean_sum_any[trait] += abs(value)
        for trait, _ in vals_any[:5]:
            top5_any[trait] += 1
        for trait, _ in vals_any[:10]:
            top10_any[trait] += 1

        # positive direction
        vals_pos = sorted(vals, key=lambda x: x[1], reverse=True)
        for trait, value in vals:
            abs_mean_sum_pos[trait] += max(value, 0.0)
        for trait, _ in vals_pos[:5]:
            top5_pos[trait] += 1
        for trait, _ in vals_pos[:10]:
            top10_pos[trait] += 1

        # negative direction
        vals_neg = sorted(vals, key=lambda x: x[1])  # most negative first
        for trait, value in vals:
            abs_mean_sum_neg[trait] += max(-value, 0.0)
        for trait, _ in vals_neg[:5]:
            top5_neg[trait] += 1
        for trait, _ in vals_neg[:10]:
            top10_neg[trait] += 1

    tables = {
        "any_top5_count": (top5_any, "top5_count"),
        "any_top10_count": (top10_any, "top10_count"),
        "any_abs_mean_sum": (abs_mean_sum_any, "abs_mean_sum"),
        "positive_top5_count": (top5_pos, "top5_count"),
        "positive_top10_count": (top10_pos, "top10_count"),
        "positive_abs_mean_sum": (abs_mean_sum_pos, "abs_mean_sum"),
        "negative_top5_count": (top5_neg, "top5_count"),
        "negative_top10_count": (top10_neg, "top10_count"),
        "negative_abs_mean_sum": (abs_mean_sum_neg, "abs_mean_sum"),
    }

    summary_json = {
        "num_traits": len(traits),
        "num_axes": len(axes),
        "tables": {},
    }

    for name, (scores, score_col) in tables.items():
        rows = to_ranked_rows(scores, score_col)
        csv_path = output_dir / f"{args.prefix}__{name}.csv"
        write_csv(csv_path, rows, ["rank", "trait", score_col])
        summary_json["tables"][name] = rows
        print(f"Saved {csv_path}")

    json_path = output_dir / f"{args.prefix}__all_tables.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

