#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

import torch

from io_utils import load_jsonl, print_header, print_kv, write_json, write_jsonl
from pipeline_utils import DEFAULT_GEN_MODEL, run_cmd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = DEFAULT_GEN_MODEL


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for repeated-run trait variation measurement."""
    parser = argparse.ArgumentParser(
        description="Run the user-trait pipeline multiple times and measure projection variation."
    )

    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--run-prefix", type=str, default=None)

    parser.add_argument("--axis", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument(
        "--generation-model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model forwarded to project/run_user_trait_pipeline.py",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization passed through to generation",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Model context length passed through to generation",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size passed through to generation",
    )

    parser.add_argument("--output-dir", type=str, required=True)

    return parser.parse_args()


def load_axis(axis_path: Path, layer: int) -> dict[str, Any]:
    """Load one layer-specific axis vector and metadata from disk."""
    data = torch.load(axis_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected axis file format: {type(data)}")

    if "vector" not in data:
        raise KeyError(f"No 'vector' key found. Available keys: {list(data.keys())}")

    vectors = data["vector"].float().cpu()

    if vectors.ndim != 2:
        raise ValueError(f"Expected 'vector' shape [n_layers, d_model], got {tuple(vectors.shape)}")

    if not (0 <= layer < vectors.shape[0]):
        raise ValueError(f"Layer {layer} out of range for vector stack {tuple(vectors.shape)}")

    return {
        "axis": vectors[layer],
        "trait": data.get("trait"),
        "activation_position": data.get("activation_position"),
        "shape": tuple(vectors.shape),
    }


def summarize(scores: list[float]) -> dict[str, Any]:
    """Compute simple descriptive statistics for a list of projection scores."""
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(scores),
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def build_pipeline_cmd(args: argparse.Namespace, run_name: str, seed: int) -> list[str]:
    """Build the single-trait pipeline command for one repeated variation run."""
    cmd = [
        "uv",
        "run",
        "python",
        "project/run_user_trait_pipeline.py",
        "--trait",
        args.trait,
        "--run-name",
        run_name,
        "--generation-model",
        args.generation_model,
        "--num-candidates",
        str(args.num_candidates),
        "--seed",
        str(seed),
        "--with-projection",
        "--axes-dir",
        str(Path(args.axis).parent),
        "--projection-mode",
        "one",
        "--axis-trait",
        Path(args.axis).stem,
        "--projection-layer",
        str(args.layer),
        "--projection-model",
        args.generation_model,
        "--max-tokens",
        "128",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]

    if args.tensor_parallel_size is not None:
        cmd += ["--tensor-parallel-size", str(args.tensor_parallel_size)]

    return cmd


def build_projection_path(trait: str, run_name: str) -> Path:
    """Return the projection-file path expected from one repeated variation run."""
    return REPO_ROOT / "outputs" / "user_prompts" / trait / "projections" / f"{run_name}.jsonl"


def main() -> None:
    """Run multiple pipelines, then score and summarize their projection variation."""
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_info = load_axis(Path(args.axis), args.layer)

    print_header("Trait Axis Variation")
    print_kv("Trait", args.trait)
    print_kv("Axis", args.axis)
    print_kv("Layer", args.layer)
    print_kv("Generation model", args.generation_model)
    print_kv("Runs", args.num_runs)
    print_kv("Candidates per intent", args.num_candidates)
    print_kv("Base seed", args.base_seed)
    print_kv("GPU memory utilization", args.gpu_memory_utilization)
    print_kv("Max model len", args.max_model_len)
    print_kv("Tensor parallel size", args.tensor_parallel_size)
    print_kv("Output dir", out_dir)

    run_prefix = args.run_prefix or f"{args.trait}_variation"
    run_records: list[dict[str, Any]] = []

    print_header("Stage 1: Pipeline Runs")
    for i in range(args.num_runs):
        seed = args.base_seed + i
        run_name = f"{run_prefix}_{i:03d}_seed_{seed}"
        projection_path = build_projection_path(args.trait, run_name)

        run_cmd(build_pipeline_cmd(args, run_name, seed), cwd=REPO_ROOT)

        if not projection_path.exists():
            raise FileNotFoundError(f"Projection file not found: {projection_path}")

        run_records.append(
            {
                "run_idx": i,
                "run_name": run_name,
                "seed": seed,
                "projection_path": projection_path,
            }
        )
        print(f"{run_name}: projection rows ready at {projection_path}")

    print_header("Stage 2: Projection Scoring")
    all_rows: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []

    for record in run_records:
        projection_rows = [
            row for row in load_jsonl(record["projection_path"])
            if row.get("projection_trait") == axis_info["trait"]
        ]
        if not projection_rows:
            raise ValueError(
                f"No projection rows found for axis trait {axis_info['trait']} in {record['projection_path']}"
            )
        run_scores: list[float] = []

        for selected_idx, row in enumerate(projection_rows):
            score = float(row["projection_score_trait"])
            run_scores.append(score)

            all_rows.append(
                {
                    "trait": args.trait,
                    "run_name": record["run_name"],
                    "run_idx": record["run_idx"],
                    "seed": record["seed"],
                    "selected_idx": selected_idx,
                    "axis_path": args.axis,
                    "axis_trait": axis_info["trait"],
                    "activation_position": axis_info["activation_position"],
                    "projection_model": args.generation_model,
                    "layer": args.layer,
                    "projection_activation_source": row.get("projection_activation_source"),
                    "neutral_response": row.get("neutral_response"),
                    "trait_response": row.get("trait_response"),
                    "projection_score": score,
                    "projection_row": row,
                }
            )

        run_summary = summarize(run_scores)
        run_summary.update(
            {
                "trait": args.trait,
                "run_name": record["run_name"],
                "run_idx": record["run_idx"],
                "seed": record["seed"],
                "num_selected": len(projection_rows),
                "axis_path": args.axis,
                "axis_trait": axis_info["trait"],
            }
        )
        per_run_rows.append(run_summary)

        mean_str = "None" if run_summary["mean"] is None else f"{run_summary['mean']:.4f}"
        std_str = "None" if run_summary["std"] is None else f"{run_summary['std']:.4f}"
        print(f"{record['run_name']}: mean={mean_str} std={std_str} n={run_summary['count']}")

    overall_scores = [row["projection_score"] for row in all_rows]
    overall_summary = summarize(overall_scores)
    overall_summary.update(
        {
            "trait": args.trait,
            "axis_path": args.axis,
            "axis_trait": axis_info["trait"],
            "activation_position": axis_info["activation_position"],
            "projection_model": args.generation_model,
            "layer": args.layer,
            "num_runs_requested": args.num_runs,
            "num_runs_completed": len(per_run_rows),
            "run_prefix": run_prefix,
            "base_seed": args.base_seed,
        }
    )

    write_jsonl(all_rows, out_dir / "all_scores.jsonl")
    write_jsonl(per_run_rows, out_dir / "per_run_summary.jsonl")
    write_json(overall_summary, out_dir / "overall_summary.json")

    print_header("Overall Summary")
    for key, value in overall_summary.items():
        print_kv(key, value)


if __name__ == "__main__":
    main()
