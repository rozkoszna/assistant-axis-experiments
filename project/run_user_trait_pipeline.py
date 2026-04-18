#!/usr/bin/env python3
"""
Run the full user-trait prompt pipeline for a single trait.

Overview
--------
This script orchestrates the end-to-end prompt-generation pipeline for one
user trait. It shells out to the stage-specific pipeline scripts, manages
output file locations, and optionally runs projection and plotting steps.

The pipeline stages are:
1. Generate candidate prompts for the requested trait.
2. Judge and score those candidates.
3. Select the best candidates according to the configured selection rules.
4. Optionally project the selected prompt pairs onto one or more saved axes.
5. Optionally plot the projection results.

Typical workflow
----------------
For a given trait (for example `confused`), the script:
- builds a run name,
- determines the standard output paths for all artifacts,
- runs generation,
- runs judging,
- runs selection,
- optionally resolves projection axes and runs projection,
- optionally creates a plot from the projection output.

Inputs
------
Required:
- `--trait`: name of the user trait to model

Optional:
- `--explanation`: additional trait description for generation
- `--run-name`: explicit run name; otherwise a generated name is used
- `--intents-file`: JSONL file of input intents

The script also accepts shared settings for:
- generation
- judging
- selection
- projection
- plotting

Projection
----------
Projection is optional and enabled with `--with-projection`.

When enabled, you must also provide:
- `--axes-dir`: directory containing saved `.pt` axis files

Projection modes:
- `all`: use all axis files in the directory
- `one`: use one named axis (requires `--axis-trait`)
- `subset`: use a subset of axes from a JSON file (requires
  `--axis-traits-file`)

Plotting
--------
Plotting is optional and enabled with `--with-plot`.

Important:
- `--with-plot` requires `--with-projection`, because the plot is created
  from the projection output.

Outputs
-------
For each run, the script writes artifacts under:
`outputs/user_prompts/<trait>/`

Including:
- `candidates/<run_name>.jsonl`
- `judged/<run_name>.jsonl`
- `selected/<run_name>.jsonl`
- `projections/<run_name>.jsonl` (if enabled)
- `plots/<run_name>.png` (if enabled)

Notes
-----
- This script is an orchestrator. It does not implement the core generation,
  judging, or selection logic itself.
- Instead, it invokes the stage-specific scripts using subprocess calls.
- It assumes the downstream scripts follow the expected file and directory
  conventions.
"""

#!/usr/bin/env python3
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
    run_cmd,
)
from projection_runner import (
    resolve_axis_files,
    run_projection_for_selected,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

def parse_args() -> argparse.Namespace:
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
    if args.with_plot and not args.with_projection:
        raise ValueError("--with-plot requires --with-projection")

    if args.with_projection and args.axes_dir is None:
        raise ValueError("--with-projection requires --axes-dir")


def print_run_summary(args: argparse.Namespace, run_name: str, paths: dict[str, Path]) -> None:
    print(f"Trait: {args.trait}")
    print(f"Run name: {run_name}")
    print(f"Candidates: {paths['candidates_file']}")
    print(f"Judged: {paths['judged_file']}")
    print(f"Selected: {paths['selected_file']}")

    if args.with_projection:
        print(f"Projections: {paths['projections_file']}")
        print(f"Projection mode: {args.projection_mode}")

    if args.with_plot:
        print(f"Plot: {paths['plot_file']}")


def build_generate_cmd(args: argparse.Namespace, candidates_file: Path) -> list[str]:
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


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_name = make_run_name(args.run_name)
    paths = build_trait_output_paths(REPO_ROOT, args.trait, run_name)

    print_run_summary(args, run_name, paths)

    gen_cmd = build_generate_cmd(args, paths["candidates_file"])
    judge_cmd = build_judge_cmd(args, paths["candidates_file"], paths["judged_file"])
    select_cmd = build_select_cmd(args, paths["judged_file"], paths["selected_file"])

    run_cmd(gen_cmd, cwd=REPO_ROOT)
    run_cmd(judge_cmd, cwd=REPO_ROOT)
    run_cmd(select_cmd, cwd=REPO_ROOT)

    if args.with_projection:
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
            selected_file=paths["selected_file"],
            output_file=paths["projections_file"],
            projection_script=REPO_ROOT / args.projection_script,
            axis_files=axis_files,
            model_name=args.projection_model,
            layer=args.projection_layer,
            run_cmd=lambda cmd: run_cmd(cmd, cwd=REPO_ROOT),
        )

    if args.with_plot:
        plot_cmd = build_plot_cmd(args, paths["projections_file"], paths["plot_file"])
        run_cmd(plot_cmd, cwd=REPO_ROOT)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
