#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from io_utils import make_run_name
from projection_runner import (
    resolve_axis_files,
    run_projection_for_selected,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_PROJECTION_MODEL = DEFAULT_GEN_MODEL
DEFAULT_INTENTS_FILE = "data/user_prompt_intents.jsonl"
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_PROJECTION_LAYER = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full user-trait prompt pipeline")

    parser.add_argument("--trait", type=str, required=True, help="Trait name, e.g. confused")
    parser.add_argument("--explanation", type=str, default=None, help="Optional trait explanation")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name")
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

    parser.add_argument("--selection-mode", choices=["best_one", "top_k", "threshold"], default="best_one")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold-score", type=float, default=85.0)
    parser.add_argument("--min-neutral-score", type=int, default=75)
    parser.add_argument("--min-trait-score", type=int, default=75)
    parser.add_argument("--min-pair-score", type=int, default=80)
    parser.add_argument("--min-final-score", type=float, default=0.0)
    parser.add_argument("--no-dedupe", action="store_true")

    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--requests-per-second", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--neutral-weight", type=float, default=0.2)
    parser.add_argument("--trait-weight", type=float, default=0.3)
    parser.add_argument("--pair-weight", type=float, default=0.5)

    parser.add_argument("--with-projection", action="store_true")
    parser.add_argument("--axes-dir", type=str, default=None, help="Directory of .pt axis files")
    parser.add_argument(
        "--projection-mode",
        choices=["all", "one", "subset"],
        default="all",
        help="Project on all axes, one axis by name, or a subset from JSON",
    )
    parser.add_argument("--axis-trait", type=str, default=None, help="Axis trait name for projection-mode=one")
    parser.add_argument(
        "--traits-file",
        type=str,
        default=None,
        help="JSON file containing a list of trait names for projection-mode=subset",
    )
    parser.add_argument("--projection-layer", type=int, default=DEFAULT_PROJECTION_LAYER)
    parser.add_argument(
        "--projection-script",
        type=str,
        default="project/project_pair_axes.py",
        help="Path to pair projection script",
    )

    parser.add_argument("--with-plot", action="store_true")
    parser.add_argument(
        "--plot-script",
        type=str,
        default="project/plots/plot_trait_run_axes.py",
        help="Path to plotting script",
    )
    parser.add_argument("--plot-top-k", type=int, default=20)

    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def validate_args(args: argparse.Namespace) -> None:
    if args.with_plot and not args.with_projection:
        raise ValueError("--with-plot requires --with-projection")

    if args.with_projection and args.axes_dir is None:
        raise ValueError("--with-projection requires --axes-dir")


def build_output_paths(trait: str, run_name: str) -> dict[str, Path]:
    trait_dir = REPO_ROOT / "outputs" / "user_prompts" / trait

    return {
        "trait_dir": trait_dir,
        "candidates_file": trait_dir / "candidates" / f"{run_name}.jsonl",
        "judged_file": trait_dir / "judged" / f"{run_name}.jsonl",
        "selected_file": trait_dir / "selected" / f"{run_name}.jsonl",
        "projections_file": trait_dir / "projections" / f"{run_name}.jsonl",
        "plot_file": trait_dir / "plots" / f"{run_name}.png",
    }


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
    paths = build_output_paths(args.trait, run_name)

    print_run_summary(args, run_name, paths)

    gen_cmd = build_generate_cmd(args, paths["candidates_file"])
    judge_cmd = build_judge_cmd(args, paths["candidates_file"], paths["judged_file"])
    select_cmd = build_select_cmd(args, paths["judged_file"], paths["selected_file"])

    run_cmd(gen_cmd)
    run_cmd(judge_cmd)
    run_cmd(select_cmd)

    if args.with_projection:
        axes_dir = REPO_ROOT / args.axes_dir
        axis_files = resolve_axis_files(
            repo_root=REPO_ROOT,
            axes_dir=axes_dir,
            projection_mode=args.projection_mode,
            axis_trait=args.axis_trait,
            traits_file=args.traits_file,
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
            run_cmd=run_cmd,
        )

    if args.with_plot:
        plot_cmd = build_plot_cmd(args, paths["projections_file"], paths["plot_file"])
        run_cmd(plot_cmd)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()