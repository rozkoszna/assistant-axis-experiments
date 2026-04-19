#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_utils import run_cmd
from projection_runner import resolve_axis_files, run_projection_for_selected


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_band(spec: str) -> tuple[float, float]:
    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"Band must look like low-high, got: {spec}")
    low, high = float(parts[0]), float(parts[1])
    if not low < high:
        raise ValueError(f"Band lower bound must be < upper bound, got: {spec}")
    return low, high


def band_label(low: float, high: float) -> str:
    return f"{low:g}-{high:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate selection, projection, and plotting for judged-score threshold bands."
    )
    parser.add_argument("--judged-file", type=str, required=True, help="Judged JSONL file to sweep")
    parser.add_argument("--analysis-name", type=str, required=True, help="Prefix used in selected/projection outputs")
    parser.add_argument("--bands", type=str, nargs="+", required=True, help="Score bands like 80-85 85-90 90-95")
    parser.add_argument("--axes-dir", type=str, required=True)
    parser.add_argument("--projection-mode", choices=["all", "one", "subset"], default="all")
    parser.add_argument("--axis-trait", type=str, default=None)
    parser.add_argument("--axis-traits-file", type=str, default=None)
    parser.add_argument("--projection-model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--projection-layer", type=int, default=20)
    parser.add_argument("--projection-script", type=str, default="project/project_pair_axes.py")
    parser.add_argument("--min-neutral-score", type=int, default=75)
    parser.add_argument("--min-trait-score", type=int, default=75)
    parser.add_argument("--min-pair-score", type=int, default=80)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--keep-empty-groups", action="store_true")
    parser.add_argument("--with-plot", action="store_true")
    parser.add_argument("--plot-output", type=str, default=None)
    parser.add_argument("--plot-top-k-axes", type=int, default=20)
    parser.add_argument("--plot-rank-by", choices=["spread", "abs_mean"], default="spread")
    parser.add_argument("--plot-title", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    judged_file = REPO_ROOT / args.judged_file
    if not judged_file.exists():
        raise FileNotFoundError(f"Judged file not found: {judged_file}")

    trait_name = judged_file.parent.parent.name
    output_root = REPO_ROOT / "outputs" / "user_prompts" / trait_name

    axis_files = resolve_axis_files(
        repo_root=REPO_ROOT,
        axes_dir=REPO_ROOT / args.axes_dir,
        projection_mode=args.projection_mode,
        axis_trait=args.axis_trait,
        traits_file=args.axis_traits_file,
    )

    projection_inputs: list[str] = []
    plot_labels: list[str] = []

    for spec in args.bands:
        low, high = parse_band(spec)
        label = band_label(low, high)
        selected_name = f"{args.analysis_name}__band_{label}.jsonl"
        selected_path = output_root / "selected" / selected_name
        projection_path = output_root / "projections" / selected_name

        select_cmd = [
            "uv",
            "run",
            "user_prompt_pipeline/3_select.py",
            "--judged_file",
            str(judged_file),
            "--output_file",
            selected_name,
            "--mode",
            "threshold",
            "--threshold_score",
            str(low),
            "--min_neutral_score",
            str(args.min_neutral_score),
            "--min_trait_score",
            str(args.min_trait_score),
            "--min_pair_score",
            str(args.min_pair_score),
            "--min_final_score",
            str(low),
            "--max_final_score",
            str(high),
        ]

        if args.no_dedupe:
            select_cmd += ["--no_dedupe"]
        if args.keep_empty_groups:
            select_cmd += ["--keep_empty_groups"]

        run_cmd(select_cmd, cwd=REPO_ROOT)

        run_projection_for_selected(
            selected_file=selected_path,
            output_file=projection_path,
            projection_script=REPO_ROOT / args.projection_script,
            axis_files=axis_files,
            model_name=args.projection_model,
            layer=args.projection_layer,
            run_cmd=lambda cmd: run_cmd(cmd, cwd=REPO_ROOT),
        )

        projection_inputs.append(str(projection_path))
        plot_labels.append(label)

    if args.with_plot:
        plot_output = args.plot_output or (
            f"outputs/analysis/{args.analysis_name}__bands.png"
        )
        plot_cmd = [
            "uv",
            "run",
            "python",
            "project/plots/plot_threshold_band_progression.py",
            "--inputs",
            *projection_inputs,
            "--labels",
            *plot_labels,
            "--output",
            plot_output,
            "--top-k-axes",
            str(args.plot_top_k_axes),
            "--rank-by",
            args.plot_rank_by,
        ]
        if args.plot_title:
            plot_cmd += ["--title", args.plot_title]
        run_cmd(plot_cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
