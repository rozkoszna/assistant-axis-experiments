#!/usr/bin/env python3
"""
Run the user-trait pipeline for multiple traits and generate comparison plots.

Overview
--------
This script orchestrates a multi-trait experiment by repeatedly running the
single-trait pipeline (`project/run_user_trait_pipeline.py`) for a list of
user traits, collecting the resulting projection files, and then generating
a combined comparison plot across those traits.

For each user trait, the script:
1. Builds a per-trait run name.
2. Runs the full single-trait pipeline with shared settings for generation,
   judging, selection, and projection.
3. Verifies that the expected projection JSONL file was created.
4. Stores that projection file for the final comparison step.

After all traits have been processed, the script:
5. Runs the appropriate comparison plotting script to visualize differences
   across traits.

Inputs
------
You must provide user traits in exactly one of these ways:
- `--user-traits`: list of trait names on the command line
- `--user-traits-file`: JSON file containing a list of trait names

You must also provide:
- `--comparison-name`: experiment name used in output filenames
- `--axes-dir`: directory containing saved `.pt` axis files

The script also accepts shared settings for:
- candidate generation
- judging
- selection
- projection
- per-trait plotting
- final comparison plotting

Projection modes
----------------
- `all`: project onto all axes in `--axes-dir`
- `one`: project onto one named axis (requires `--axis-trait`)
- `subset`: project onto a subset of axes from a JSON file
  (requires `--axis-traits-file`)

Outputs
-------
Per trait:
- Runs the standard single-trait pipeline and produces the usual per-trait
  outputs, including a projection JSONL file.

Across all traits:
- Produces one comparison plot in `--comparison-output-dir`.

Plotting behavior
-----------------
- If `--with-per-trait-plots` is enabled, each single-trait run also produces
  its own standard plot.
- Independently of that, this script always produces a final comparison plot
  across all requested traits.

Notes
-----
- This script is an orchestration wrapper. It does not directly implement
  generation, judging, selection, or projection logic itself.
- Instead, it launches the single-trait pipeline as a subprocess for each
  trait and then launches a comparison plotting script.
- The script assumes the standard output path convention used by
  `run_user_trait_pipeline.py`.
"""

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_PROJECTION_MODEL = DEFAULT_GEN_MODEL
DEFAULT_INTENTS_FILE = "data/user_prompt_intents.jsonl"
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_PROJECTION_LAYER = 20


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--intents-file", type=str, default=DEFAULT_INTENTS_FILE)
    parser.add_argument("--generation-model", type=str, default=DEFAULT_GEN_MODEL)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--projection-model", type=str, default=DEFAULT_PROJECTION_MODEL)

    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-intents", action="store_true")

    # Selection args
    parser.add_argument("--selection-mode", choices=["best_one", "top_k", "threshold"], default="best_one")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold-score", type=float, default=85.0)
    parser.add_argument("--min-neutral-score", type=int, default=75)
    parser.add_argument("--min-trait-score", type=int, default=75)
    parser.add_argument("--min-pair-score", type=int, default=80)
    parser.add_argument("--min-final-score", type=float, default=0.0)
    parser.add_argument("--no-dedupe", action="store_true")

    # Judge args
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--requests-per-second", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--neutral-weight", type=float, default=0.2)
    parser.add_argument("--trait-weight", type=float, default=0.3)
    parser.add_argument("--pair-weight", type=float, default=0.5)

    # Projection args
    parser.add_argument("--axes-dir", type=str, required=True, help="Directory of .pt axis files")
    parser.add_argument(
        "--projection-mode",
        choices=["all", "one", "subset"],
        default="all",
        help="Project on all axes, one axis by name, or a subset from JSON",
    )
    parser.add_argument("--axis-trait", type=str, default=None, help="Axis trait name for projection-mode=one")
    parser.add_argument(
        "--axis-traits-file",
        type=str,
        default=None,
        help="JSON file containing a list of axis trait names for projection-mode=subset",
    )
    parser.add_argument("--projection-layer", type=int, default=DEFAULT_PROJECTION_LAYER)
    parser.add_argument(
        "--projection-script",
        type=str,
        default="project/project_pair_axes.py",
        help="Path to pair projection script",
    )

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


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_trait_list(path: Path) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{path} must contain a JSON list of strings")

    return data


def get_user_traits(args: argparse.Namespace) -> list[str]:
    if args.user_traits is not None:
        return args.user_traits
    return load_trait_list(REPO_ROOT / args.user_traits_file)


def build_trait_run_name(trait: str, comparison_name: str, run_suffix: str | None) -> str:
    parts = [comparison_name, trait]
    if run_suffix:
        parts.append(run_suffix)
    return "__".join(parts)


def build_single_trait_cmd(args: argparse.Namespace, trait: str, run_name: str) -> list[str]:
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


def get_projection_file(trait: str, run_name: str) -> Path:
    return REPO_ROOT / "outputs" / "user_prompts" / trait / "projections" / f"{run_name}.jsonl"


def build_comparison_plot_cmd(
    args: argparse.Namespace,
    projection_files: list[Path],
) -> list[str]:
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
        run_cmd(cmd)

        projection_file = get_projection_file(trait, run_name)
        if not projection_file.exists():
            raise FileNotFoundError(f"Expected projection file not found: {projection_file}")

        projection_files.append(projection_file)

    print("\nProjection files:")
    for path in projection_files:
        print(f"  - {path}")

    comparison_cmd = build_comparison_plot_cmd(args, projection_files)
    run_cmd(comparison_cmd)

    print("\nMulti-trait analysis finished successfully.")


if __name__ == "__main__":
    main()