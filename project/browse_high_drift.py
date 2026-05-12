#!/usr/bin/env python3
"""
Browse responses with the highest drift for specific (user_trait, axis) pairs.

Reads from projections JSONL files (which already contain the merged response
text) and prints the top-K examples sorted by |projection_delta_trait_minus_neutral|.

Usage
-----
# Show top 3 examples for each of the hardcoded pairs:
uv run browse_high_drift.py \
    --projections-dir outputs/user_prompts \
    --run-prefix strict_all_axes_llama_100eval \
    --top-k 3

# Override with custom pairs (user_trait:axis):
uv run browse_high_drift.py \
    --projections-dir outputs/user_prompts \
    --run-prefix strict_all_axes_llama_100eval \
    --pairs skeptical:condescending empathetic:emotional \
    --top-k 5

# Truncate long responses:
uv run browse_high_drift.py ... --max-chars 600
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Top 10 pairs from significance analysis (user_trait, axis, direction)
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("skeptical",    "condescending"),
    ("pessimistic",  "entertaining"),
    ("empathetic",   "emotional"),
    ("empathetic",   "dispassionate"),
    ("empathetic",   "empathetic"),
    ("skeptical",    "entertaining"),
    ("anxious",      "emotional"),
    ("anxious",      "entertaining"),
    ("empathetic",   "entertaining"),
    ("skeptical",    "impatient"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse highest-drift response pairs")
    parser.add_argument(
        "--projections-dir",
        required=True,
        help="Root dir containing per-trait subfolders, e.g. outputs/user_prompts",
    )
    parser.add_argument(
        "--run-prefix",
        required=True,
        help="Run name prefix, e.g. strict_all_axes_llama_100eval",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="TRAIT:AXIS",
        help="Pairs to inspect (default: top-10 hardcoded pairs). Format: user_trait:axis",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top examples to show per pair (default: 3)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Max characters to print per response (default: 500)",
    )
    parser.add_argument(
        "--show-neutral",
        action="store_true",
        help="Also print the neutral prompt and response (default: off)",
    )
    return parser.parse_args()


def find_projections_file(projections_dir: Path, user_trait: str, run_prefix: str) -> Path | None:
    trait_dir = projections_dir / user_trait / "projections"
    if not trait_dir.exists():
        return None
    candidates = [
        f for f in trait_dir.glob(f"{run_prefix}*.jsonl")
        if not f.name.endswith("__neutral.jsonl")
    ]
    if not candidates:
        return None
    # Prefer exact prefix match, then longest name
    candidates.sort(key=lambda p: (not p.stem.startswith(run_prefix), -len(p.stem)))
    return candidates[0]


def load_rows_for_axis(proj_file: Path, axis: str) -> list[dict]:
    rows = []
    with open(proj_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("projection_trait", "")) == axis:
                rows.append(row)
    return rows


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"  [...{len(text) - max_chars} more chars]"


def print_pair_examples(
    user_trait: str,
    axis: str,
    rows: list[dict],
    top_k: int,
    max_chars: int,
    show_neutral: bool,
) -> None:
    if not rows:
        print(f"  [no rows found for axis '{axis}']")
        return

    rows_sorted = sorted(rows, key=lambda r: abs(float(r["projection_delta_trait_minus_neutral"])), reverse=True)
    n = min(top_k, len(rows_sorted))

    # Print column header for delta values in the block
    deltas = [float(r["projection_delta_trait_minus_neutral"]) for r in rows_sorted]
    mean_delta = sum(deltas) / len(deltas)
    print(f"  n={len(rows)}  mean_delta={mean_delta:+.3f}  showing top {n} by |delta|")

    for rank, row in enumerate(rows_sorted[:n], 1):
        delta = float(row["projection_delta_trait_minus_neutral"])
        score_t = float(row["projection_score_trait"])
        score_n = float(row["projection_score_neutral"])
        trait_prompt = row.get("trait_prompt", row.get("prompt_b", "?"))
        trait_response = row.get("trait_response", row.get("response_b", "?"))
        neutral_prompt = row.get("neutral_prompt", row.get("prompt_a", "?"))
        neutral_response = row.get("neutral_response", row.get("response_a", "?"))

        print(f"\n  [{rank}] delta={delta:+.3f}  score_trait={score_t:.3f}  score_neutral={score_n:.3f}")
        print(f"  TRAIT PROMPT:    {truncate(trait_prompt, max_chars)}")
        print(f"  TRAIT RESPONSE:  {truncate(trait_response, max_chars)}")
        if show_neutral:
            print(f"  NEUTRAL PROMPT:  {truncate(neutral_prompt, max_chars)}")
            print(f"  NEUTRAL RESPONSE:{truncate(neutral_response, max_chars)}")
    print()


def main() -> None:
    args = parse_args()
    proj_dir = Path(args.projections_dir)

    if args.pairs:
        pairs = []
        for p in args.pairs:
            if ":" not in p:
                raise ValueError(f"Pair must be in format TRAIT:AXIS, got: {p!r}")
            t, a = p.split(":", 1)
            pairs.append((t.strip(), a.strip()))
    else:
        pairs = DEFAULT_PAIRS

    for user_trait, axis in pairs:
        print("=" * 72)
        print(f"USER TRAIT: {user_trait}   AXIS: {axis}")
        print("=" * 72)

        proj_file = find_projections_file(proj_dir, user_trait, args.run_prefix)
        if proj_file is None:
            print(f"  [projections file not found under {proj_dir / user_trait / 'projections'}]")
            print()
            continue

        rows = load_rows_for_axis(proj_file, axis)
        print_pair_examples(user_trait, axis, rows, args.top_k, args.max_chars, args.show_neutral)


if __name__ == "__main__":
    main()
