#!/usr/bin/env python3
"""
Select the best judged USER prompt pairs.

This script reads judged rows from 2_judge.py and keeps:
- the single best row per intent, or
- the top-k rows per intent, or
- all rows above a threshold per intent.

It also supports basic deduplication inside each intent group.

Example:
    uv run user_prompt_pipeline/3_select.py \
      --judged_file outputs/user_prompts/confused/judged.jsonl \
      --output_file outputs/user_prompts/confused/selected.jsonl \
      --mode best_one

    uv run user_prompt_pipeline/3_select.py \
      --judged_file outputs/user_prompts/confused/judged.jsonl \
      --output_file outputs/user_prompts/confused/selected_top3.jsonl \
      --mode top_k \
      --top_k 3
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jsonlines


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class JudgedCandidatePair:
    trait: str
    explanation: Optional[str]
    intent_index: int
    intent: str
    candidate_index: int
    neutral_prompt: str
    trait_prompt: str
    neutral_score: int
    trait_score: int
    pair_score: int
    final_score: float


def load_judged(path: Path) -> list[JudgedCandidatePair]:
    rows: list[JudgedCandidatePair] = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            rows.append(JudgedCandidatePair(**row))
    return rows


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`]+", "", text)
    return text


def pair_signature(row: JudgedCandidatePair) -> tuple[str, str]:
    return (
        normalize_text(row.neutral_prompt),
        normalize_text(row.trait_prompt),
    )


def passes_thresholds(
    row: JudgedCandidatePair,
    min_neutral_score: int,
    min_trait_score: int,
    min_pair_score: int,
    min_final_score: float,
) -> bool:
    return (
        row.neutral_score >= min_neutral_score
        and row.trait_score >= min_trait_score
        and row.pair_score >= min_pair_score
        and row.final_score >= min_final_score
    )


def dedupe_rows(rows: list[JudgedCandidatePair]) -> list[JudgedCandidatePair]:
    seen: set[tuple[str, str]] = set()
    deduped: list[JudgedCandidatePair] = []

    for row in rows:
        sig = pair_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(row)

    return deduped


def group_by_intent(rows: list[JudgedCandidatePair]) -> dict[int, list[JudgedCandidatePair]]:
    grouped: dict[int, list[JudgedCandidatePair]] = defaultdict(list)
    for row in rows:
        grouped[row.intent_index].append(row)
    return dict(grouped)


def sort_group(rows: list[JudgedCandidatePair]) -> list[JudgedCandidatePair]:
    return sorted(
        rows,
        key=lambda row: (
            row.final_score,
            row.pair_score,
            row.trait_score,
            row.neutral_score,
            -row.candidate_index,
        ),
        reverse=True,
    )


def select_from_group(
    rows: list[JudgedCandidatePair],
    mode: str,
    top_k: int,
    threshold_score: float,
) -> list[JudgedCandidatePair]:
    if mode == "best_one":
        return rows[:1]

    if mode == "top_k":
        return rows[:top_k]

    if mode == "threshold":
        return [row for row in rows if row.final_score >= threshold_score]

    raise ValueError(f"Unknown mode: {mode}")


def select_rows(
    rows: list[JudgedCandidatePair],
    mode: str,
    top_k: int,
    threshold_score: float,
    min_neutral_score: int,
    min_trait_score: int,
    min_pair_score: int,
    min_final_score: float,
    dedupe: bool,
    keep_empty_groups: bool,
) -> list[JudgedCandidatePair]:
    grouped = group_by_intent(rows)
    selected: list[JudgedCandidatePair] = []

    num_groups = len(grouped)
    empty_after_filter = 0

    for intent_index, group in grouped.items():
        filtered = [
            row
            for row in group
            if passes_thresholds(
                row=row,
                min_neutral_score=min_neutral_score,
                min_trait_score=min_trait_score,
                min_pair_score=min_pair_score,
                min_final_score=min_final_score,
            )
        ]

        filtered = sort_group(filtered)

        if dedupe:
            filtered = dedupe_rows(filtered)

        picked = select_from_group(
            rows=filtered,
            mode=mode,
            top_k=top_k,
            threshold_score=threshold_score,
        )

        if not picked:
            empty_after_filter += 1
            if keep_empty_groups:
                logger.warning(
                    "No rows survived for intent_index=%s after filtering/selection",
                    intent_index,
                )
            continue

        selected.extend(picked)

    logger.info("Intent groups processed: %s", num_groups)
    logger.info("Intent groups with no surviving rows: %s", empty_after_filter)
    return selected


def save_selected(path: Path, rows: list[JudgedCandidatePair]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(asdict(row))
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best judged USER prompt pairs")
    parser.add_argument("--judged_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["best_one", "top_k", "threshold"],
        default="best_one",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument(
        "--threshold_score",
        type=float,
        default=85.0,
        help="Used only in threshold mode: keep rows with final_score >= this value",
    )
    parser.add_argument("--min_neutral_score", type=int, default=75)
    parser.add_argument("--min_trait_score", type=int, default=75)
    parser.add_argument("--min_pair_score", type=int, default=80)
    parser.add_argument("--min_final_score", type=float, default=0.0)
    parser.add_argument("--no_dedupe", action="store_true")
    parser.add_argument("--keep_empty_groups", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "top_k" and args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    judged_file = Path(args.judged_file)
    output_file = Path(args.output_file)

    rows = load_judged(judged_file)
    logger.info("Loaded %s judged rows from %s", len(rows), judged_file)

    selected = select_rows(
        rows=rows,
        mode=args.mode,
        top_k=args.top_k,
        threshold_score=args.threshold_score,
        min_neutral_score=args.min_neutral_score,
        min_trait_score=args.min_trait_score,
        min_pair_score=args.min_pair_score,
        min_final_score=args.min_final_score,
        dedupe=not args.no_dedupe,
        keep_empty_groups=args.keep_empty_groups,
    )

    count = save_selected(output_file, selected)
    logger.info("Saved %s selected rows to %s", count, output_file)


if __name__ == "__main__":
    main()
