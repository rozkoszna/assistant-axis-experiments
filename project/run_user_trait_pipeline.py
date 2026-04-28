#!/usr/bin/env python3
"""
Run the full user-trait experiment for one trait.

What this script does
---------------------
This is the main orchestrator for experiments like:
- "What happens when the user sounds confused?"
- "How does the assistant change for a very confused user?"

One run always means:
- one user trait, passed with `--trait`
- many prompt/response examples for that trait
- optionally, projection onto one or more assistant axes

The pipeline
------------
For one trait, the script runs these stages in order:

1. Generate matched user prompt pairs:
   - `neutral_prompt`
   - `trait_prompt`

2. Judge and filter those prompt pairs.

3. Keep the selected prompt pairs.

4. If projection is enabled:
   - generate assistant answers for both sides
   - save:
     - `neutral_response`
     - `trait_response`
   - save generation-time assistant activations in:
     - `activations/<run_name>.pt`

5. Project the saved assistant-side activations onto one or more precomputed
   assistant axes.

6. Optionally make a plot from the projection output.

What is being measured
----------------------
This script does not project user prompts.

The intended measured object is the assistant response:
- neutral user prompt -> neutral assistant response
- trait user prompt -> trait assistant response

Projection uses the saved generation-time activations from the response stage.
If the activation file is missing, projection should stop with an error.

Neutral vs trait
----------------
Each selected example is a matched pair:
- neutral side
- trait-conditioned side

So the "neutral" output is not a separate experiment. It is the baseline side
of the same run, for the same selected examples.

That is why the pipeline can also save:
- `projections/<run_name>__neutral.jsonl`

This file is a reusable neutral baseline for the same run across the same
assistant axes.

Outputs
-------
Artifacts are written under:
- `outputs/user_prompts/<trait>/`

Main files:
- `candidates/<run_name>.jsonl`
- `judged/<run_name>.jsonl`
- `selected/<run_name>.jsonl`
- `responses/<run_name>.jsonl`
- `activations/<run_name>.pt`
- `projections/<run_name>.jsonl`
- `projections/<run_name>__neutral.jsonl`
- `plots/<run_name>.png`

Important options
-----------------
- `--with-projection` enables response generation, activation capture, and projection.
- `--with-plot` requires `--with-projection`.
- `--axes-dir` points to the saved `.pt` assistant axes.
- `--projection-mode` controls whether projection uses:
  - all axes
  - one named axis
  - a subset of axes from a file

This script is an orchestrator only. The actual generation, judging,
activation capture, and projection logic live in the stage-specific files it
calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import make_run_name
from pipeline_utils import (
    add_generation_args,
    add_judge_args,
    add_plot_args,
    add_projection_args,
    add_selection_args,
    build_trait_output_paths,
    resolve_axis_files,
    run_cmd,
)
from projection_runner import run_projection_for_selected


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the single-trait user-prompt pipeline."""
    parser = argparse.ArgumentParser(description="Run the full user-trait prompt pipeline")

    parser.add_argument("--trait", type=str, required=True, help="Trait name, e.g. confused")
    parser.add_argument("--explanation", type=str, default=None, help="Optional trait explanation")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name")
    add_generation_args(parser)
    add_selection_args(parser)
    add_judge_args(parser)
    add_projection_args(parser, axes_dir_required=False, include_projection_toggle=True)
    add_plot_args(parser)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate combinations of projection and plotting arguments."""
    if args.with_plot and not args.with_projection:
        raise ValueError("--with-plot requires --with-projection")

    if args.with_projection and args.axes_dir is None:
        raise ValueError("--with-projection requires --axes-dir")


def print_run_summary(args: argparse.Namespace, run_name: str, paths: dict[str, Path]) -> None:
    """Print the resolved output files and optional projection/plot settings."""
    print(f"Trait: {args.trait}")
    print(f"Run name: {run_name}")
    print(f"Candidates: {paths['candidates_file']}")
    print(f"Judged: {paths['judged_file']}")
    print(f"Selected: {paths['selected_file']}")

    if args.with_projection:
        print(f"Responses: {paths['responses_file']}")
        print(f"Activations: {paths['activations_file']}")
        print(f"Projections: {paths['projections_file']}")
        print(f"Neutral baseline: {paths['neutral_projections_file']}")
        print(f"Projection mode: {args.projection_mode}")

    if args.with_plot:
        print(f"Plot: {paths['plot_file']}")


def build_generate_cmd(args: argparse.Namespace, candidates_file: Path) -> list[str]:
    """Build the stage-1 command that generates candidate prompt pairs."""
    cmd = [
        "uv",
        "run",
        "user_prompt_pipeline/1_generate.py",
        "--model",
        args.generation_model,
        "--trait",
        args.trait,
        "--intents_file",
        args.intents_file,
        "--output_file",
        candidates_file.name,
        "--num_candidates",
        str(args.num_candidates),
        "--generation_batch_size",
        str(args.generation_batch_size),
        "--temperature",
        str(args.temperature),
        "--max_tokens",
        str(args.max_tokens),
        "--top_p",
        str(args.top_p),
        "--max_model_len",
        str(args.max_model_len),
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.explanation:
        cmd += ["--explanation", args.explanation]
    if args.tensor_parallel_size is not None:
        cmd += ["--tensor_parallel_size", str(args.tensor_parallel_size)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.shuffle_intents:
        cmd += ["--shuffle_intents"]

    return cmd


def build_judge_cmd(args: argparse.Namespace, candidates_file: Path, judged_file: Path) -> list[str]:
    """Build the stage-2 command that scores candidate prompt pairs."""
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
    """Build the stage-3 command that filters judged prompt pairs."""
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


def build_plot_cmd(args: argparse.Namespace, projections_file: Path, plot_file: Path) -> list[str]:
    """Build the optional plotting command for one run's projection output."""
    return [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / args.plot_script),
        "--input",
        str(projections_file),
        "--output",
        str(plot_file),
        "--top-k",
        str(args.plot_top_k),
    ]


def build_response_cmd(
    args: argparse.Namespace,
    selected_file: Path,
    responses_file: Path,
    activations_file: Path,
) -> list[str]:
    """Build the stage-4 command that generates assistant responses for selected prompts."""
    cmd = [
        "uv",
        "run",
        "user_prompt_pipeline/4_generate_responses.py",
        "--selected-file",
        str(selected_file),
        "--output-file",
        str(responses_file),
        "--activations-file",
        str(activations_file),
        "--model",
        args.projection_model,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.max_tokens),
        "--top-p",
        str(args.top_p),
        "--temperature",
        str(args.response_temperature),
    ]

    if args.tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]

    return cmd


def main() -> None:
    """Run the full single-trait pipeline from prompt generation through plotting."""
    args = parse_args()
    validate_args(args)

    # Resolve one stable run name and the canonical output paths used by all stages.
    run_name = make_run_name(args.run_name)
    paths = build_trait_output_paths(REPO_ROOT, args.trait, run_name)

    print_run_summary(args, run_name, paths)

    # Build the three prompt-side stages first: generate -> judge -> select.
    gen_cmd = build_generate_cmd(args, paths["candidates_file"])
    judge_cmd = build_judge_cmd(args, paths["candidates_file"], paths["judged_file"])
    select_cmd = build_select_cmd(args, paths["judged_file"], paths["selected_file"])

    # Execute the prompt-side pipeline in order so each stage can consume the
    # artifact produced by the previous one.
    run_cmd(gen_cmd, cwd=REPO_ROOT)
    run_cmd(judge_cmd, cwd=REPO_ROOT)
    run_cmd(select_cmd, cwd=REPO_ROOT)

    if args.with_projection:
        # Before projection, generate assistant responses to the selected prompt
        # pairs. This keeps the final measurement focused on model behavior rather
        # than just prompt wording.
        response_cmd = build_response_cmd(
            args,
            paths["selected_file"],
            paths["responses_file"],
            paths["activations_file"],
        )
        run_cmd(response_cmd, cwd=REPO_ROOT)

        # Resolve which saved axes to use, then project the saved assistant
        # activations onto those axes.
        axes_dir = REPO_ROOT / args.axes_dir
        axis_files = resolve_axis_files(
            repo_root=REPO_ROOT,
            axes_dir=axes_dir,
            projection_mode=args.projection_mode,
            axis_trait=args.axis_trait,
            traits_file=args.axis_traits_file,
        )

        print("Resolved axis files:")
        for axis_file in axis_files:
            print(f"  - {axis_file}")

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

    if args.with_plot:
        # Plot only after projection exists, since the plotter reads the saved
        # projection JSONL rather than regenerating anything itself.
        plot_cmd = build_plot_cmd(args, paths["projections_file"], paths["plot_file"])
        run_cmd(plot_cmd, cwd=REPO_ROOT)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
