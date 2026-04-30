#!/usr/bin/env python3
"""
Judge generated USER prompt pairs.

For each candidate pair from 1_generate.py, this script produces:
- neutral_score: how good the neutral prompt is
- trait_score: how well the trait prompt expresses the requested trait
- pair_score: how well the two prompts form a matched style-only contrastive pair
- final_score: weighted aggregate used later by 3_select.py

Example:
    uv run user_prompt_pipeline/2_judge.py \
      --candidates_file outputs/user_prompts/confused/candidates.jsonl \
      --output_file outputs/user_prompts/confused/judged.jsonl \
      --judge_model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import jsonlines
import openai
from dotenv import load_dotenv

# Reuse the repo's judge helpers.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from assistant_axis.judge import RateLimiter, call_judge_batch
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import assistant_axis.judge helpers. "
        "Make sure you run this from the repository root with dependencies installed."
    ) from exc


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@dataclass
class CandidatePair:
    trait: str
    explanation: Optional[str]
    intent_index: int
    topic: Optional[str] = None
    intent: str
    candidate_index: int
    neutral_prompt: str
    trait_prompt: str


@dataclass
class JudgedCandidatePair:
    trait: str
    explanation: Optional[str]
    intent_index: int
    topic: Optional[str] = None
    intent: str
    candidate_index: int
    neutral_prompt: str
    trait_prompt: str
    neutral_score: int
    trait_score: int
    pair_score: int
    final_score: float


def load_candidates(path: Path) -> list[CandidatePair]:
    """Load generated candidate prompt pairs from JSONL."""
    rows: list[CandidatePair] = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            rows.append(CandidatePair(**row))
    return rows


def build_explanation_block(explanation: Optional[str]) -> str:
    """Render optional trait explanation text for judge prompts."""
    if explanation:
        return f"Explanation:\n{explanation}\n"
    return "Use a common-sense interpretation of the trait.\n"


def build_neutral_judge_prompt(intent: str, prompt: str) -> str:
    """Build judge prompt for neutral quality scoring."""
    return f"""You are evaluating whether a generated USER prompt is a good neutral version of a seed intent.

Seed intent:
{intent}

Generated USER prompt:
{prompt}

Return REFUSAL if:
- this is not a user prompt
- it answers the request instead of asking it
- it is malformed, empty, or nonsensical

Otherwise return a single integer from 0 to 100.

Score high only if:
- it clearly preserves the seed intent
- it is neutral, clear, direct, and natural
- it does not add a strong persona, emotion, or stylistic flavor
- it sounds like something a real user would write

Guidance:
90-100 = excellent neutral prompt, fully faithful to intent
70-89 = good neutral prompt, minor issues
40-69 = partially neutral or somewhat awkward
10-39 = weak, stylized, or semantically off
0-9 = wrong mode, malformed, or does not preserve intent
"""


def build_trait_judge_prompt(
    trait: str,
    explanation: Optional[str],
    intent: str,
    prompt: str,
) -> str:
    """Build judge prompt for trait-expression scoring."""
    explanation_block = build_explanation_block(explanation)
    return f"""You are evaluating whether a generated USER prompt expresses a requested trait while preserving the seed intent.

Trait:
{trait}

{explanation_block}
Seed intent:
{intent}

Generated USER prompt:
{prompt}

Return REFUSAL if:
- this is not a user prompt
- it answers the request instead of asking it
- it is malformed, empty, or nonsensical

Otherwise return a single integer from 0 to 100.

Score high only if:
- it clearly expresses the requested trait
- it preserves the seed intent
- it sounds natural and plausible as something a real user would write
- the trait is expressed mainly through style/tone, not by changing the meaning

Guidance:
90-100 = strong trait expression, natural, faithful to intent
70-89 = good trait expression, small issues
40-69 = weak trait expression or some semantic drift
10-39 = mostly wrong style or awkward
0-9 = wrong mode, malformed, or wrong intent
"""

def build_pair_judge_prompt(
    trait: str,
    explanation: Optional[str],
    intent: str,
    neutral_prompt: str,
    trait_prompt: str,
) -> str:
    """Build judge prompt for paired contrast quality scoring."""
    explanation_block = build_explanation_block(explanation)
    return f"""You are evaluating a pair of USER prompts generated from the same seed intent.

Requested trait:
{trait}

{explanation_block}
Seed intent:
{intent}

Neutral prompt:
{neutral_prompt}

Trait prompt:
{trait_prompt}

Return REFUSAL if either prompt:
- is not a user prompt
- answers the request instead of asking it
- is malformed, empty, or nonsensical

Otherwise return ONLY a single integer from 0 to 100.
Do not include any explanation, labels, punctuation, or extra words.
Examples of valid outputs:
92
7
100

Score high only if:
- both prompts preserve the same seed intent
- the neutral prompt is genuinely neutral
- the trait prompt clearly expresses the requested trait
- the main difference between the two prompts is style/tone/persona, not meaning
- both prompts sound natural and plausible

Guidance:
90-100 = excellent pair, same intent, strong clean contrast
70-89 = good pair, mostly same intent, minor issues
40-69 = noticeable semantic drift or weak contrast
10-39 = poor pairing, major mismatch
0-9 = malformed, wrong mode, or clearly different intents
"""


def parse_judge_score(response_text: Optional[str]) -> int:
    """Parse integer score from judge response text."""
    if not response_text:
        return 0

    text = response_text.strip()
    if text.upper().startswith("REFUSAL"):
        return 0

    match = re.search(r"\b(100|[1-9]?\d)\b", text)
    if not match:
        return 0

    score = int(match.group(1))
    return score if 0 <= score <= 100 else 0


async def judge_rows(
    rows: list[CandidatePair],
    judge_model: str,
    max_tokens: int,
    requests_per_second: int,
    batch_size: int,
    neutral_weight: float,
    trait_weight: float,
    pair_weight: float,
) -> list[JudgedCandidatePair]:
    """Run batched judging and return rows with merged scores."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(requests_per_second)

    neutral_prompts = [
        build_neutral_judge_prompt(row.intent, row.neutral_prompt)
        for row in rows
    ]
    trait_prompts = [
        build_trait_judge_prompt(row.trait, row.explanation, row.intent, row.trait_prompt)
        for row in rows
    ]
    pair_prompts = [
        build_pair_judge_prompt(
            row.trait,
            row.explanation,
            row.intent,
            row.neutral_prompt,
            row.trait_prompt,
        )
        for row in rows
    ]

    logger.info("Judging neutral prompts: %s rows", len(rows))
    neutral_raw = await call_judge_batch(
        client=client,
        prompts=neutral_prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )

    logger.info("Judging trait prompts: %s rows", len(rows))
    trait_raw = await call_judge_batch(
        client=client,
        prompts=trait_prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )

    logger.info("Judging prompt pairs: %s rows", len(rows))
    pair_raw = await call_judge_batch(
        client=client,
        prompts=pair_prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )

    judged: list[JudgedCandidatePair] = []
    for row, n_raw, t_raw, p_raw in zip(rows, neutral_raw, trait_raw, pair_raw):
        neutral_score = parse_judge_score(n_raw)
        trait_score = parse_judge_score(t_raw)
        pair_score = parse_judge_score(p_raw)

        final_score = (
            neutral_weight * neutral_score
            + trait_weight * trait_score
            + pair_weight * pair_score
        )

        judged.append(
            JudgedCandidatePair(
                trait=row.trait,
                explanation=row.explanation,
                intent_index=row.intent_index,
                topic=row.topic,
                intent=row.intent,
                candidate_index=row.candidate_index,
                neutral_prompt=row.neutral_prompt,
                trait_prompt=row.trait_prompt,
                neutral_score=neutral_score,
                trait_score=trait_score,
                pair_score=pair_score,
                final_score=round(final_score, 4),
            )
        )

    return judged


def save_judged(path: Path, rows: list[JudgedCandidatePair]) -> int:
    """Save judged rows to JSONL and return written count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(asdict(row))
    return len(rows)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for judging stage."""
    parser = argparse.ArgumentParser(description="Judge generated USER prompt pairs")
    parser.add_argument("--candidates_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--requests_per_second", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--neutral_weight", type=float, default=0.2)
    parser.add_argument("--trait_weight", type=float, default=0.3)
    parser.add_argument("--pair_weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run the judging stage end to end."""
    args = parse_args()

    total_weight = args.neutral_weight + args.trait_weight + args.pair_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight} "
            f"({args.neutral_weight}, {args.trait_weight}, {args.pair_weight})"
        )

    candidates_file = Path(args.candidates_file)
    trait_name = candidates_file.parent.parent.name
    base_dir = Path("outputs") / "user_prompts" / trait_name
    output_file = base_dir / "judged" / Path(args.output_file).name

    rows = load_candidates(candidates_file)
    logger.info("Loaded %s candidate pairs from %s", len(rows), candidates_file)

    judged = asyncio.run(
        judge_rows(
            rows=rows,
            judge_model=args.judge_model,
            max_tokens=args.max_tokens,
            requests_per_second=args.requests_per_second,
            batch_size=args.batch_size,
            neutral_weight=args.neutral_weight,
            trait_weight=args.trait_weight,
            pair_weight=args.pair_weight,
        )
    )

    count = save_judged(output_file, judged)
    logger.info("Saved %s judged rows to %s", count, output_file)


if __name__ == "__main__":
    main()
