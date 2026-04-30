#!/usr/bin/env python3
"""
Generate candidate USER prompt pairs for a given trait and a bank of seed intents.

For each intent, this script generates:
- one or more neutral USER prompts
- one or more trait-expressing USER prompts

The output is a JSONL file of candidate prompt pairs that can later be judged
by an LLM and filtered down to the best examples.

Example:
    uv run user_prompt_pipeline/1_generate.py \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --trait confused \
      --explanation "sounds unsure, hesitant, slightly rambly" \
      --intents_file data/user_prompt_intents.jsonl \
      --output_file outputs/user_prompts/confused/candidates.jsonl \
      --num_candidates 5
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import jsonlines
import torch

# Reuse the repo's generation backend.
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from assistant_axis.generation import VLLMGenerator
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Could not import assistant_axis.generation.VLLMGenerator. "
        "Make sure you run this from the repository root with dependencies installed."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = "You generate realistic USER prompts only."


@dataclass
class IntentRecord:
    intent_index: int
    topic: Optional[str]
    intent: str


@dataclass
class CandidatePair:
    trait: str
    explanation: Optional[str]
    intent_index: int
    topic: Optional[str]
    intent: str
    candidate_index: int
    neutral_prompt: str
    trait_prompt: str


def load_intents(path: Path) -> list[IntentRecord]:
    intents: list[IntentRecord] = []
    with jsonlines.open(path, "r") as reader:
        for idx, row in enumerate(reader):
            if isinstance(row, str):
                intent = row.strip()
            elif isinstance(row, dict):
                intent = str(row.get("intent") or row.get("question") or "").strip()
                topic = row.get("topic")
                topic = str(topic).strip() if topic is not None else None
            else:
                intent = ""
                topic = None

            if not intent:
                logger.warning("Skipping empty intent at row %s", idx)
                continue

            intents.append(IntentRecord(intent_index=idx, topic=topic, intent=intent))

    if not intents:
        raise ValueError(f"No valid intents found in {path}")

    return intents


def build_style_block(trait: str, explanation: Optional[str]) -> str:
    if explanation:
        return (
            f"Trait: {trait}\n"
            f"Explanation: {explanation}\n"
            "Use a natural, common-sense interpretation of the trait. "
            "Do not exaggerate it into parody unless the seed intent strongly supports that."
        )
    return (
        f"Trait: {trait}\n"
        "Use a natural, common-sense interpretation of the trait. "
        "Do not exaggerate it into parody unless the seed intent strongly supports that."
    )


def build_neutral_user_prompt(intent: str) -> str:
    return (
        "Generate exactly one USER prompt.\n\n"
        "Requirements:\n"
        "- Do not answer the request.\n"
        "- Do not act as the assistant.\n"
        "- Preserve the seed intent exactly.\n"
        "- Output only the user prompt text.\n"
        "- Write in a clear, neutral, direct, natural way.\n"
        "- Do not add any strong persona or stylistic flavor.\n\n"
        f"Seed intent:\n{intent}\n"
    )


def build_trait_user_prompt(
    intent: str,
    trait: str,
    explanation: Optional[str],
    neutral_reference: Optional[str] = None,
    stronger_contrast: bool = False,
) -> str:
    style_block = build_style_block(trait, explanation)
    contrast_block = ""
    if neutral_reference:
        contrast_block = (
            "Reference neutral USER prompt (preserve meaning, but do not copy wording):\n"
            f"{neutral_reference}\n\n"
            "Contrast requirements:\n"
            "- Keep the same underlying request and constraints.\n"
            "- Change wording and sentence rhythm clearly.\n"
            "- Express the requested trait through style/tone.\n"
            "- Do not copy the opening phrase from the reference prompt.\n\n"
        )
    if stronger_contrast:
        contrast_block += (
            "Stronger contrast requirement:\n"
            "- Make the style difference obvious at first glance while preserving meaning.\n"
            "- Avoid reusing distinctive phrases from the reference prompt.\n\n"
        )

    return (
        "Generate exactly one USER prompt.\n\n"
        "Requirements:\n"
        "- Do not answer the request.\n"
        "- Do not act as the assistant.\n"
        "- Preserve the seed intent exactly.\n"
        "- Output only the user prompt text.\n"
        "- The difference from a neutral version should be mostly style, not meaning.\n"
        "- Keep it natural and plausible as something a real user would type.\n\n"
        f"Seed intent:\n{intent}\n\n"
        f"{contrast_block}"
        f"Style:\n{style_block}\n"
    )


def sanitize_output(text: str) -> str:
    text = text.strip()

    # Remove surrounding quotes if the model wraps the prompt.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()

    # Keep only the first non-empty line if the model starts adding commentary.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    # If the first line looks like a label, try the next line.
    if lines[0].lower().startswith(("user prompt:", "prompt:", "output:")) and len(lines) > 1:
        return lines[1]

    return lines[0]


def lexical_similarity(a: str, b: str) -> float:
    left = " ".join(a.lower().split())
    right = " ".join(b.lower().split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(a=left, b=right).ratio()


class PromptPairGenerator:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        temperature: float,
        max_tokens: int,
        top_p: float,
        seed: Optional[int],
    ) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed
        self.generator = None

    def load(self) -> None:
        self.generator = VLLMGenerator(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        self.generator.load()

    def generate_many(self, user_prompts: list[str]) -> list[str]:
        if self.generator is None:
            raise RuntimeError("Generator is not loaded")

        if not user_prompts:
            return []

        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            for user_prompt in user_prompts
        ]
        outputs = self.generator.generate_batch(conversations)
        if len(outputs) != len(user_prompts):
            raise RuntimeError(f"Expected {len(user_prompts)} outputs, got {len(outputs)}")
        return [sanitize_output(str(text)) for text in outputs]

    def generate_one(self, user_prompt: str) -> str:
        outputs = self.generate_many([user_prompt])
        return outputs[0] if outputs else ""


def compute_tensor_parallel_size(user_value: Optional[int]) -> int:
    if user_value is not None:
        return user_value

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
        return max(1, len(visible))

    count = torch.cuda.device_count()
    return max(1, count)


def iter_candidate_pairs(
    generator: PromptPairGenerator,
    intents: Iterable[IntentRecord],
    trait: str,
    explanation: Optional[str],
    num_candidates: int,
    shuffle_intents: bool,
    generation_batch_size: int,
) -> Iterable[CandidatePair]:
    if generation_batch_size < 1:
        raise ValueError("generation_batch_size must be >= 1")

    intent_list = list(intents)
    if shuffle_intents:
        random.shuffle(intent_list)

    neutral_jobs: list[tuple[IntentRecord, int, str]] = []
    for item in intent_list:
        for candidate_index in range(num_candidates):
            neutral_instruction = build_neutral_user_prompt(item.intent)
            neutral_jobs.append((item, candidate_index, neutral_instruction))

    total_pairs = len(neutral_jobs)
    logger.info("Generating %s neutral prompts in batches of %s", total_pairs, generation_batch_size)

    neutral_by_key: dict[tuple[int, int], str] = {}
    for start in range(0, total_pairs, generation_batch_size):
        chunk = neutral_jobs[start : start + generation_batch_size]
        instructions = [instruction for _, _, instruction in chunk]
        responses = generator.generate_many(instructions)
        logger.info("Generated neutral prompts %s/%s", min(start + len(chunk), total_pairs), total_pairs)
        for (item, candidate_index, _), response in zip(chunk, responses):
            neutral_by_key[(item.intent_index, candidate_index)] = response

    trait_jobs: list[tuple[IntentRecord, int, str]] = []
    for item in intent_list:
        for candidate_index in range(num_candidates):
            key = (item.intent_index, candidate_index)
            neutral_reference = neutral_by_key.get(key, "")
            trait_instruction = build_trait_user_prompt(
                item.intent,
                trait,
                explanation,
                neutral_reference=neutral_reference,
            )
            trait_jobs.append((item, candidate_index, trait_instruction))

    logger.info("Generating %s trait prompts in batches of %s", total_pairs, generation_batch_size)
    trait_by_key: dict[tuple[int, int], str] = {}
    for start in range(0, total_pairs, generation_batch_size):
        chunk = trait_jobs[start : start + generation_batch_size]
        instructions = [instruction for _, _, instruction in chunk]
        responses = generator.generate_many(instructions)
        logger.info("Generated trait prompts %s/%s", min(start + len(chunk), total_pairs), total_pairs)
        for (item, candidate_index, _), response in zip(chunk, responses):
            trait_by_key[(item.intent_index, candidate_index)] = response

    similar_pairs: list[tuple[IntentRecord, int, str]] = []
    for item in intent_list:
        for candidate_index in range(num_candidates):
            key = (item.intent_index, candidate_index)
            neutral_prompt = neutral_by_key.get(key, "")
            trait_prompt = trait_by_key.get(key, "")
            if lexical_similarity(neutral_prompt, trait_prompt) >= 0.9:
                retry_instruction = build_trait_user_prompt(
                    item.intent,
                    trait,
                    explanation,
                    neutral_reference=neutral_prompt,
                    stronger_contrast=True,
                )
                similar_pairs.append((item, candidate_index, retry_instruction))

    if similar_pairs:
        logger.info("Retrying %s overly-similar trait prompts for stronger contrast", len(similar_pairs))
        for start in range(0, len(similar_pairs), generation_batch_size):
            chunk = similar_pairs[start : start + generation_batch_size]
            instructions = [instruction for _, _, instruction in chunk]
            responses = generator.generate_many(instructions)
            for (item, candidate_index, _), response in zip(chunk, responses):
                trait_by_key[(item.intent_index, candidate_index)] = response

    for item in intent_list:
        for candidate_index in range(num_candidates):
            key = (item.intent_index, candidate_index)
            neutral_prompt = neutral_by_key.get(key, "")
            trait_prompt = trait_by_key.get(key, "")

            if not neutral_prompt:
                logger.warning(
                    "Empty neutral prompt for intent_index=%s candidate_index=%s",
                    item.intent_index,
                    candidate_index,
                )
            if not trait_prompt:
                logger.warning(
                    "Empty trait prompt for intent_index=%s candidate_index=%s",
                    item.intent_index,
                    candidate_index,
                )

            yield CandidatePair(
                trait=trait,
                explanation=explanation,
                intent_index=item.intent_index,
                topic=item.topic,
                intent=item.intent,
                candidate_index=candidate_index,
                neutral_prompt=neutral_prompt,
                trait_prompt=trait_prompt,
            )


def save_candidates(path: Path, rows: Iterable[CandidatePair]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(asdict(row))
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate USER prompt pairs for a trait")
    parser.add_argument("--model", type=str, required=True, help="Model name for vLLM generation")
    parser.add_argument("--trait", type=str, required=True, help="Trait name, e.g. confused")
    parser.add_argument(
        "--explanation",
        type=str,
        default=None,
        help="Optional short explanation refining the trait",
    )
    parser.add_argument(
        "--intents_file",
        type=str,
        required=True,
        help="JSONL file with {'intent': ...} rows",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to save candidate prompt pairs as JSONL",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=5,
        help="How many neutral/trait pairs to generate per intent",
    )
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=256,
        help="Number of prompt-generation requests to submit per vLLM batch call",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle_intents", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    intents_file = Path(args.intents_file)
    base_dir = Path("outputs") / "user_prompts" / args.trait
    output_file = base_dir / "candidates" / Path(args.output_file).name

    intents = load_intents(intents_file)
    logger.info("Loaded %s intents from %s", len(intents), intents_file)
    logger.info("Resolved output file to %s", output_file)

    tensor_parallel_size = compute_tensor_parallel_size(args.tensor_parallel_size)

    generator = PromptPairGenerator(
        model_name=args.model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seed,
    )
    logger.info("Loading generator for model=%s", args.model)
    generator.load()

    rows = iter_candidate_pairs(
        generator=generator,
        intents=intents,
        trait=args.trait,
        explanation=args.explanation,
        num_candidates=args.num_candidates,
        shuffle_intents=args.shuffle_intents,
        generation_batch_size=args.generation_batch_size,
    )
    count = save_candidates(output_file, rows)

    logger.info("Saved %s candidate pairs to %s", count, output_file)


if __name__ == "__main__":
    main()
