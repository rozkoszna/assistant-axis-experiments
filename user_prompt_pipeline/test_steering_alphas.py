#!/usr/bin/env python3
"""
Test steered prompt generation at different alpha values.

Generates neutral + trait prompts at multiple alpha strengths for a handful
of intents and saves results to a JSON file for easy inspection.

Usage:
    uv run user_prompt_pipeline/test_steering_alphas.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --vector precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs/entertaining.pt \
        --trait entertaining \
        --intents-file data/identity_probe_intents.jsonl \
        --alphas 0 10 20 30 \
        --n-intents 3 \
        --output outputs/steering_test/entertaining_alphas.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "generate",
    os.path.join(Path(__file__).parent, "1_generate.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_neutral_user_prompt = _mod.build_neutral_user_prompt
build_trait_user_prompt = _mod.build_trait_user_prompt
load_intents = _mod.load_intents
sanitize_output = _mod.sanitize_output
SYSTEM_PROMPT = _mod.SYSTEM_PROMPT
from persona_steering.steer import SteeredModel, parse_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vector", required=True, help="Path to .pt axis vector")
    parser.add_argument("--trait", required=True)
    parser.add_argument("--explanation", default=None)
    parser.add_argument("--intents-file", required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0, 10, 20, 30])
    parser.add_argument("--n-intents", type=int, default=3, help="Number of intents to test")
    parser.add_argument("--layers", type=str, default="13-22")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    intents = load_intents(Path(args.intents_file))[: args.n_intents]
    layer = parse_layers(args.layers)

    # Load model once, change alpha per run
    model = SteeredModel.from_pretrained(
        model_name=args.model,
        vector_path=args.vector,
        alpha=args.alphas[0],
        layer=layer,
    )

    results = []

    for intent in intents:
        neutral_instruction = build_neutral_user_prompt(intent.intent)
        trait_instruction = build_trait_user_prompt(intent.intent, args.trait, args.explanation)

        entry = {
            "intent_index": intent.intent_index,
            "topic": intent.topic,
            "intent": intent.intent,
            "neutral": None,
            "trait_by_alpha": {},
        }

        # Neutral: always unsteered
        neutral_out = model.generate_unsteered(
            neutral_instruction,
            system_prompt=SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        entry["neutral"] = sanitize_output(neutral_out)
        print(f"\n[intent {intent.intent_index}] neutral: {entry['neutral'][:120]}")

        # Trait at each alpha
        for alpha in args.alphas:
            model.alpha = alpha
            if alpha == 0:
                out = model.generate_unsteered(
                    trait_instruction,
                    system_prompt=SYSTEM_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            else:
                out = model.generate(
                    trait_instruction,
                    system_prompt=SYSTEM_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            text = sanitize_output(out)
            entry["trait_by_alpha"][str(alpha)] = text
            print(f"[intent {intent.intent_index}] alpha={alpha:5.1f}: {text[:120]}")

        results.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "trait": args.trait,
            "vector": args.vector,
            "layers": args.layers,
            "alphas": args.alphas,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
