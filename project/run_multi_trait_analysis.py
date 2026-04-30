#!/usr/bin/env python3
"""
Run multi-trait user-prompt experiments and aggregate projection outputs.

This script supports two execution modes:
- default subprocess mode: call the single-trait orchestrator per trait
- reuse mode (`--reuse-models`): keep stage-1 and stage-4 models loaded across
  traits to avoid repeated model initialization overhead

Projection in both modes uses the assistant model's internal vectors captured
while generating responses (`answer_mean` hidden states). It does not project
the raw prompt/response text itself.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
from typing import Any

import jsonlines
import torch

from io_utils import load_trait_list
from pipeline_utils import (
    add_generation_args,
    add_judge_args,
    add_projection_args,
    add_selection_args,
    build_trait_output_paths,
    get_projection_output_path,
    resolve_axis_files,
    run_cmd,
)
from projection_runner import run_projection_for_selected

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))
from assistant_axis.generation import VLLMGenerator  # noqa: E402
from assistant_axis.internals import ProbingModel  # noqa: E402


SYSTEM_PROMPT = "You generate realistic USER prompts only."


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for multi-trait analysis."""
    parser = argparse.ArgumentParser(
        description="Run user-trait pipeline for many traits and generate comparison plots."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-traits", nargs="+", type=str)
    group.add_argument("--user-traits-file", type=str)

    parser.add_argument("--comparison-name", type=str, required=True)
    parser.add_argument("--run-suffix", type=str, default=None)
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reuse loaded stage-1 and stage-4 models across all traits in one process",
    )

    add_generation_args(parser)
    add_selection_args(parser)
    add_judge_args(parser)
    add_projection_args(parser, axes_dir_required=True, include_projection_toggle=False)

    parser.add_argument("--with-per-trait-plots", action="store_true")
    parser.add_argument("--plot-top-k", type=int, default=20)
    parser.add_argument("--comparison-output-dir", type=str, default="outputs/analysis")
    parser.add_argument("--aggregate", choices=["mean", "median"], default="mean")
    parser.add_argument("--top-k-axes", type=int, default=20)
    return parser.parse_args()


def get_user_traits(args: argparse.Namespace) -> list[str]:
    """Resolve trait names from CLI list or JSON file."""
    if args.user_traits is not None:
        return args.user_traits
    return load_trait_list(REPO_ROOT / args.user_traits_file)


def build_trait_run_name(trait: str, comparison_name: str, run_suffix: str | None) -> str:
    """Build a stable per-trait run name."""
    parts = [comparison_name, trait]
    if run_suffix:
        parts.append(run_suffix)
    return "__".join(parts)


def normalize_text(text: str) -> str:
    """Normalize text for lexical similarity checks."""
    return " ".join(text.lower().split())


def lexical_similarity(a: str, b: str) -> float:
    """Return sequence-level lexical similarity ratio in [0, 1]."""
    return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def build_style_block(trait: str) -> str:
    """Build trait guidance block used during trait prompt generation."""
    return (
        f"Trait: {trait}\n"
        "Use a natural, common-sense interpretation of the trait. "
        "Do not exaggerate it into parody unless the seed intent strongly supports that."
    )


def build_neutral_user_prompt(intent: str) -> str:
    """Build neutral user-prompt generation instruction for one intent."""
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
    neutral_reference: str,
    *,
    stronger_contrast: bool,
) -> str:
    """Build trait-conditioned user-prompt instruction for one intent."""
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
        f"Style:\n{build_style_block(trait)}\n"
    )


def sanitize_output(text: str) -> str:
    """Extract one clean user prompt line from model output."""
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if lines[0].lower().startswith(("user prompt:", "prompt:", "output:")) and len(lines) > 1:
        return lines[1]
    return lines[0]


def load_intents(path: Path) -> list[dict[str, Any]]:
    """Load and normalize seed intents from JSONL."""
    rows: list[dict[str, Any]] = []
    with jsonlines.open(path, "r") as reader:
        for idx, row in enumerate(reader):
            if isinstance(row, str):
                intent = row.strip()
                topic = None
            elif isinstance(row, dict):
                intent = str(row.get("intent") or row.get("question") or "").strip()
                topic = row.get("topic")
                topic = str(topic).strip() if topic is not None else None
            else:
                intent = ""
                topic = None
            if intent:
                rows.append({"intent_index": idx, "topic": topic, "intent": intent})
    if not rows:
        raise ValueError(f"No valid intents found in {path}")
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to JSONL and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def build_generation_conversations(prompts: list[str]) -> list[list[dict[str, str]]]:
    """Wrap raw prompts in system+user chat conversations."""
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]


def generate_candidates_for_trait(
    *,
    generator: VLLMGenerator,
    trait: str,
    intents: list[dict[str, Any]],
    num_candidates: int,
    generation_batch_size: int,
    output_file: Path,
) -> None:
    """Generate candidate neutral/trait prompt pairs for one trait."""
    jobs: list[tuple[int, int, str]] = []
    for item in intents:
        for candidate_index in range(num_candidates):
            jobs.append((item["intent_index"], candidate_index, item["intent"]))

    neutral_by_key: dict[tuple[int, int], str] = {}
    for start in range(0, len(jobs), generation_batch_size):
        chunk = jobs[start : start + generation_batch_size]
        prompts = [build_neutral_user_prompt(intent) for _, _, intent in chunk]
        outputs = generator.generate_batch(build_generation_conversations(prompts))
        for (intent_index, candidate_index, _), text in zip(chunk, outputs):
            neutral_by_key[(intent_index, candidate_index)] = sanitize_output(str(text))

    trait_by_key: dict[tuple[int, int], str] = {}
    for start in range(0, len(jobs), generation_batch_size):
        chunk = jobs[start : start + generation_batch_size]
        prompts = []
        for intent_index, candidate_index, intent in chunk:
            neutral = neutral_by_key[(intent_index, candidate_index)]
            prompts.append(build_trait_user_prompt(intent, trait, neutral, stronger_contrast=False))
        outputs = generator.generate_batch(build_generation_conversations(prompts))
        for (intent_index, candidate_index, _), text in zip(chunk, outputs):
            trait_by_key[(intent_index, candidate_index)] = sanitize_output(str(text))

    retry_jobs: list[tuple[int, int, str]] = []
    for intent_index, candidate_index, intent in jobs:
        key = (intent_index, candidate_index)
        if lexical_similarity(neutral_by_key[key], trait_by_key[key]) >= 0.9:
            retry_jobs.append((intent_index, candidate_index, intent))

    for start in range(0, len(retry_jobs), generation_batch_size):
        chunk = retry_jobs[start : start + generation_batch_size]
        prompts = []
        for intent_index, candidate_index, intent in chunk:
            neutral = neutral_by_key[(intent_index, candidate_index)]
            prompts.append(build_trait_user_prompt(intent, trait, neutral, stronger_contrast=True))
        outputs = generator.generate_batch(build_generation_conversations(prompts))
        for (intent_index, candidate_index, _), text in zip(chunk, outputs):
            trait_by_key[(intent_index, candidate_index)] = sanitize_output(str(text))

    intent_lookup = {item["intent_index"]: item["intent"] for item in intents}
    topic_lookup = {item["intent_index"]: item.get("topic") for item in intents}
    rows: list[dict[str, Any]] = []
    for intent_index, candidate_index, _ in jobs:
        key = (intent_index, candidate_index)
        rows.append(
            {
                "trait": trait,
                "explanation": None,
                "intent_index": intent_index,
                "topic": topic_lookup.get(intent_index),
                "intent": intent_lookup[intent_index],
                "candidate_index": candidate_index,
                "neutral_prompt": neutral_by_key.get(key, ""),
                "trait_prompt": trait_by_key.get(key, ""),
            }
        )
    save_jsonl(output_file, rows)


def build_chat_kwargs(model_name: str) -> dict[str, Any]:
    """Return optional chat-template kwargs for specific model families."""
    if "qwen" in model_name.lower():
        return {"enable_thinking": False}
    return {}


def generate_response_with_answer_mean(
    pm: ProbingModel,
    conversation: list[dict[str, str]],
    *,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, torch.Tensor]:
    """Generate one response and compute per-layer mean over answer tokens."""
    tokenizer = pm.tokenizer
    model = pm.model
    device = pm.device
    chat_kwargs = build_chat_kwargs(pm.model_name)
    formatted_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    max_prompt_tokens = max(1, max_model_len - max_new_tokens)
    prompt_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    input_ids = prompt_inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    model_layers = pm.get_layers()
    num_layers = len(model_layers)
    captured_sequence: dict[int, torch.Tensor] = {}
    handles = []

    def create_hook_fn(layer_idx: int):
        def hook_fn(module, inputs, output):
            act_tensor = output[0] if isinstance(output, tuple) else output
            captured_sequence[layer_idx] = act_tensor[0].detach().cpu()

        return hook_fn

    for layer_idx, layer_module in enumerate(model_layers):
        handles.append(layer_module.register_forward_hook(create_hook_fn(layer_idx)))

    try:
        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": prompt_inputs["attention_mask"].to(device),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
            full_input_ids = output_ids[:, :max_model_len]
            _ = model(full_input_ids)
    finally:
        for handle in handles:
            handle.remove()

    generated_ids = full_input_ids[0, prompt_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if generated_ids.numel() == 0:
        hidden_size = model.config.hidden_size
        return response_text, torch.zeros((num_layers, hidden_size), dtype=torch.bfloat16)

    answer_activations = []
    for layer_idx in range(num_layers):
        layer_seq = captured_sequence[layer_idx]
        answer_activations.append(layer_seq[prompt_len:, :].mean(dim=0))
    return response_text, torch.stack(answer_activations).to(torch.bfloat16)


def generate_responses_for_trait(
    *,
    pm: ProbingModel,
    selected_file: Path,
    responses_file: Path,
    activations_file: Path,
    max_model_len: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    save_every: int,
) -> None:
    """Generate assistant responses + activation payload for selected rows."""
    selected_rows: list[dict[str, Any]] = []
    with jsonlines.open(selected_file, "r") as reader:
        for row in reader:
            selected_rows.append(dict(row))

    responses: list[dict[str, Any]] = []
    activations: list[dict[str, Any]] = []
    total = len(selected_rows)
    for idx, row in enumerate(selected_rows, start=1):
        neutral_response, neutral_answer_mean = generate_response_with_answer_mean(
            pm,
            [{"role": "user", "content": row["neutral_prompt"]}],
            max_model_len=max_model_len,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        trait_response, trait_answer_mean = generate_response_with_answer_mean(
            pm,
            [{"role": "user", "content": row["trait_prompt"]}],
            max_model_len=max_model_len,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        enriched = dict(row)
        enriched["neutral_response"] = neutral_response
        enriched["trait_response"] = trait_response
        responses.append(enriched)
        activations.append(
            {
                "neutral_answer_mean": neutral_answer_mean,
                "trait_answer_mean": trait_answer_mean,
            }
        )
        if idx % save_every == 0 or idx == total:
            save_jsonl(responses_file, responses)
            activations_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "activation_position": "answer_mean",
                    "model_name": pm.model_name,
                    "rows": activations,
                },
                activations_file,
            )


def build_judge_cmd(args: argparse.Namespace, candidates_file: Path, judged_file: Path) -> list[str]:
    """Build stage-2 judge command."""
    return [
        "uv",
        "run",
        "user_prompt_pipeline/2_judge.py",
        "--candidates_file",
        str(candidates_file),
        "--output_file",
        judged_file.name,
        "--judge_model",
        args.judge_model,
        "--max_tokens",
        str(args.judge_max_tokens),
        "--requests_per_second",
        str(args.requests_per_second),
        "--batch_size",
        str(args.batch_size),
        "--neutral_weight",
        str(args.neutral_weight),
        "--trait_weight",
        str(args.trait_weight),
        "--pair_weight",
        str(args.pair_weight),
    ]


def build_select_cmd(args: argparse.Namespace, judged_file: Path, selected_file: Path) -> list[str]:
    """Build stage-3 selection command."""
    cmd = [
        "uv",
        "run",
        "user_prompt_pipeline/3_select.py",
        "--judged_file",
        str(judged_file),
        "--output_file",
        selected_file.name,
        "--mode",
        args.selection_mode,
        "--top_k",
        str(args.top_k),
        "--threshold_score",
        str(args.threshold_score),
        "--min_neutral_score",
        str(args.min_neutral_score),
        "--min_trait_score",
        str(args.min_trait_score),
        "--min_pair_score",
        str(args.min_pair_score),
        "--min_final_score",
        str(args.min_final_score),
    ]
    if args.no_dedupe:
        cmd += ["--no_dedupe"]
    return cmd


def build_per_trait_plot_cmd(args: argparse.Namespace, projection_file: Path, output_file: Path) -> list[str]:
    """Build optional per-trait plotting command."""
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "project/plots/plot_trait_run_axes.py"),
        "--input",
        str(projection_file),
        "--output",
        str(output_file),
        "--top-k",
        str(args.plot_top_k),
    ]


def build_comparison_plot_cmd(args: argparse.Namespace, projection_files: list[Path]) -> list[str]:
    """Build final cross-trait comparison plotting command."""
    output_dir = REPO_ROOT / args.comparison_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.projection_mode == "one":
        output_path = output_dir / f"{args.comparison_name}__one_axis__{args.axis_trait}.png"
        return [
            "uv",
            "run",
            "python",
            str(REPO_ROOT / "project/plots/plot_many_traits_one_axis.py"),
            "--inputs",
            *[str(p) for p in projection_files],
            "--axis-trait",
            args.axis_trait,
            "--aggregate",
            args.aggregate,
            "--output",
            str(output_path),
        ]
    output_path = output_dir / f"{args.comparison_name}__many_axes.png"
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "project/plots/plot_many_traits_many_axes.py"),
        "--inputs",
        *[str(p) for p in projection_files],
        "--aggregate",
        args.aggregate,
        "--top-k-axes",
        str(args.top_k_axes),
        "--output",
        str(output_path),
    ]


def run_reuse_pipeline(args: argparse.Namespace, traits: list[str]) -> list[Path]:
    """Run all traits in-process while reusing loaded stage-1/stage-4 models."""
    intents = load_intents(REPO_ROOT / args.intents_file)
    axis_files = resolve_axis_files(
        repo_root=REPO_ROOT,
        axes_dir=REPO_ROOT / args.axes_dir,
        projection_mode=args.projection_mode,
        axis_trait=args.axis_trait,
        traits_file=args.axis_traits_file,
    )

    generator = VLLMGenerator(
        model_name=args.generation_model,
        tensor_parallel_size=args.tensor_parallel_size or max(1, torch.cuda.device_count()),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    generator.load()
    pm = ProbingModel(args.projection_model)

    projection_files: list[Path] = []
    try:
        for trait in traits:
            run_name = build_trait_run_name(trait, args.comparison_name, args.run_suffix)
            paths = build_trait_output_paths(REPO_ROOT, trait, run_name)

            print(f"\n=== Trait: {trait} ({run_name}) ===")
            generate_candidates_for_trait(
                generator=generator,
                trait=trait,
                intents=intents,
                num_candidates=args.num_candidates,
                generation_batch_size=args.generation_batch_size,
                output_file=paths["candidates_file"],
            )
            run_cmd(build_judge_cmd(args, paths["candidates_file"], paths["judged_file"]), cwd=REPO_ROOT)
            run_cmd(build_select_cmd(args, paths["judged_file"], paths["selected_file"]), cwd=REPO_ROOT)
            generate_responses_for_trait(
                pm=pm,
                selected_file=paths["selected_file"],
                responses_file=paths["responses_file"],
                activations_file=paths["activations_file"],
                max_model_len=args.max_model_len,
                max_tokens=args.max_tokens,
                temperature=args.response_temperature,
                top_p=args.top_p,
                save_every=10,
            )
            run_projection_for_selected(
                selected_file=paths["responses_file"],
                activations_file=paths["activations_file"],
                output_file=paths["projections_file"],
                neutral_output_file=paths["neutral_projections_file"],
                projection_script=REPO_ROOT / args.projection_script,
                axis_files=axis_files,
                model_name=args.projection_model,
                layer=args.projection_layer,
                run_cmd=lambda cmd: run_cmd(cmd, cwd=REPO_ROOT),
            )
            if args.with_per_trait_plots:
                run_cmd(
                    build_per_trait_plot_cmd(args, paths["projections_file"], paths["plot_file"]),
                    cwd=REPO_ROOT,
                )
            projection_files.append(paths["projections_file"])
    finally:
        pm.close()

    return projection_files


def run_subprocess_pipeline(args: argparse.Namespace, traits: list[str]) -> list[Path]:
    """Run all traits by invoking the single-trait pipeline subprocess."""
    projection_files: list[Path] = []
    for trait in traits:
        run_name = build_trait_run_name(trait=trait, comparison_name=args.comparison_name, run_suffix=args.run_suffix)
        cmd = [
            "uv",
            "run",
            "python",
            "project/run_user_trait_pipeline.py",
            "--trait",
            trait,
            "--run-name",
            run_name,
            "--intents-file",
            args.intents_file,
            "--generation-model",
            args.generation_model,
            "--judge-model",
            args.judge_model,
            "--projection-model",
            args.projection_model,
            "--num-candidates",
            str(args.num_candidates),
            "--generation-batch-size",
            str(args.generation_batch_size),
            "--temperature",
            str(args.temperature),
            "--response-temperature",
            str(args.response_temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--top-p",
            str(args.top_p),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--selection-mode",
            args.selection_mode,
            "--top-k",
            str(args.top_k),
            "--threshold-score",
            str(args.threshold_score),
            "--min-neutral-score",
            str(args.min_neutral_score),
            "--min-trait-score",
            str(args.min_trait_score),
            "--min-pair-score",
            str(args.min_pair_score),
            "--min-final-score",
            str(args.min_final_score),
            "--judge-max-tokens",
            str(args.judge_max_tokens),
            "--requests-per-second",
            str(args.requests_per_second),
            "--batch-size",
            str(args.batch_size),
            "--neutral-weight",
            str(args.neutral_weight),
            "--trait-weight",
            str(args.trait_weight),
            "--pair-weight",
            str(args.pair_weight),
            "--with-projection",
            "--axes-dir",
            args.axes_dir,
            "--projection-mode",
            args.projection_mode,
            "--projection-layer",
            str(args.projection_layer),
            "--projection-script",
            args.projection_script,
        ]
        if args.axis_trait is not None:
            cmd += ["--axis-trait", args.axis_trait]
        if args.axis_traits_file is not None:
            cmd += ["--axis-traits-file", args.axis_traits_file]
        if args.tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.shuffle_intents:
            cmd += ["--shuffle-intents"]
        if args.no_dedupe:
            cmd += ["--no-dedupe"]
        if args.with_per_trait_plots:
            cmd += ["--with-plot", "--plot-top-k", str(args.plot_top_k)]

        run_cmd(cmd, cwd=REPO_ROOT)
        projection_file = get_projection_output_path(REPO_ROOT, trait, run_name)
        if not projection_file.exists():
            raise FileNotFoundError(f"Expected projection file not found: {projection_file}")
        projection_files.append(projection_file)
    return projection_files


def main() -> None:
    """Run multi-trait pipeline and generate the comparison plot."""
    args = parse_args()
    traits = get_user_traits(args)
    if not traits:
        raise ValueError("No traits provided")

    print("Traits to run:")
    for trait in traits:
        print(f"  - {trait}")

    projection_files = run_reuse_pipeline(args, traits) if args.reuse_models else run_subprocess_pipeline(args, traits)

    print("\nProjection files:")
    for path in projection_files:
        print(f"  - {path}")

    run_cmd(build_comparison_plot_cmd(args, projection_files), cwd=REPO_ROOT)
    print("\nMulti-trait analysis finished successfully.")


if __name__ == "__main__":
    main()
