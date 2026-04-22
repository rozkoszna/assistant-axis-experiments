#!/usr/bin/env python3
"""
Run the user-trait experiment for several traits and compare them.

What this script does
---------------------
This is the multi-trait wrapper around `project/run_user_trait_pipeline.py`.

Use it for experiments like:
- "Compare confused vs very_confused"
- "Run several user traits on the same subset of assistant axes"

How it works
------------
For each requested user trait, this script:
- builds a per-trait run name
- runs the full single-trait pipeline
- collects the resulting projection file

After all traits finish, it makes one comparison plot from those projection
files.

One important mental model
--------------------------
Each inner run still means:
- one user trait
- many selected examples for that trait
- one or more assistant axes

So this script adds a second layer:
- many user traits across runs
- many assistant axes within each run

That is why one projection file can already contain many axis rows:
- the user trait is fixed for that file
- the assistant axis changes row by row via `projection_trait`

Neutral side
------------
Each per-trait run also contains a matched neutral side for the same selected
examples.

So files like:
- `projections/<run_name>.jsonl`
- `projections/<run_name>__neutral.jsonl`

still belong to one user-trait run. They do not mix traits together. The
`__neutral` file is just the neutral baseline view of that same run.

Inputs
------
You provide the user traits either with:
- `--user-traits`
- `--user-traits-file`

You also provide:
- `--comparison-name`
- `--axes-dir`

The rest of the options are shared settings forwarded into the single-trait
pipeline.

Outputs
-------
For each trait, this script produces the usual single-trait outputs, including:
- responses
- activations
- projections

Then it creates one final comparison plot in:
- `--comparison-output-dir`

Important options
-----------------
- `--projection-mode` controls whether each trait run uses:
  - all axes
  - one axis
  - a subset of axes
- `--with-per-trait-plots` also keeps the individual plots from each inner run

This script is only an orchestrator. The actual generation, response capture,
activation saving, and projection work happens inside
`project/run_user_trait_pipeline.py`.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import load_trait_list
from pipeline_utils import (
    add_generation_args,
    add_judge_args,
    add_projection_args,
    add_selection_args,
    get_projection_output_path,
    run_cmd,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the multi-trait comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Run user-trait pipeline for many traits and generate comparison plots."
    )

    # Multi-trait input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--user-traits",
        nargs="+",
        type=str,
        help="List of USER traits to run, e.g. confused very_very_confused",
    )
    group.add_argument(
        "--user-traits-file",
        type=str,
        help="JSON file containing a list of USER trait names",
    )

    parser.add_argument(
        "--comparison-name",
        type=str,
        required=True,
        help="Name for this multi-trait experiment, used in output filenames",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to each per-trait run-name",
    )

    # Shared pipeline args
    add_generation_args(parser)

    # Selection args
    add_selection_args(parser)

    # Judge args
    add_judge_args(parser)

    # Projection args
    add_projection_args(parser, axes_dir_required=True, include_projection_toggle=False)

    # Per-run plot
    parser.add_argument(
        "--with-per-trait-plots",
        action="store_true",
        help="Also create the standard per-trait run plot for each run",
    )
    parser.add_argument("--plot-top-k", type=int, default=20)

    # Comparison plot outputs
    parser.add_argument(
        "--comparison-output-dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save multi-trait comparison outputs",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation over rows",
    )
    parser.add_argument(
        "--top-k-axes",
        type=int,
        default=20,
        help="For many-axis comparison plots, keep top K axes",
    )

    return parser.parse_args()
def get_user_traits(args: argparse.Namespace) -> list[str]:
    """Resolve the trait list from CLI values or a JSON file."""
    if args.user_traits is not None:
        return args.user_traits
    return load_trait_list(REPO_ROOT / args.user_traits_file)


def build_trait_run_name(trait: str, comparison_name: str, run_suffix: str | None) -> str:
    """Construct the per-trait run name used for downstream artifacts."""
    parts = [comparison_name, trait]
    if run_suffix:
        parts.append(run_suffix)
    return "__".join(parts)


def build_single_trait_cmd(args: argparse.Namespace, trait: str, run_name: str) -> list[str]:
    """Build the command that launches the single-trait pipeline for one trait."""
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
        "--temperature",
        str(args.temperature),
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

    return cmd


def build_comparison_plot_cmd(
    args: argparse.Namespace,
    projection_files: list[Path],
) -> list[str]:
    """Build the final comparison-plot command based on projection mode."""
    output_dir = REPO_ROOT / args.comparison_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.projection_mode == "one":
        if not args.axis_trait:
            raise ValueError("--projection-mode one requires --axis-trait")

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


def main() -> None:
    """Run the single-trait pipeline for each trait and then create a comparison plot."""
    args = parse_args()

    traits = get_user_traits(args)
    if not traits:
        raise ValueError("No traits provided")

    print("Traits to run:")
    for trait in traits:
        print(f"  - {trait}")

    projection_files: list[Path] = []

    for trait in traits:
        run_name = build_trait_run_name(
            trait=trait,
            comparison_name=args.comparison_name,
            run_suffix=args.run_suffix,
        )

        cmd = build_single_trait_cmd(args, trait, run_name)
        run_cmd(cmd, cwd=REPO_ROOT)

        projection_file = get_projection_output_path(REPO_ROOT, trait, run_name)
        if not projection_file.exists():
            raise FileNotFoundError(f"Expected projection file not found: {projection_file}")

        projection_files.append(projection_file)

    print("\nProjection files:")
    for path in projection_files:
        print(f"  - {path}")

    comparison_cmd = build_comparison_plot_cmd(args, projection_files)
    run_cmd(comparison_cmd, cwd=REPO_ROOT)

    print("\nMulti-trait analysis finished successfully.")


if __name__ == "__main__":
    main()
