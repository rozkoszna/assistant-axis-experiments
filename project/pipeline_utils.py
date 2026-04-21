from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_PROJECTION_MODEL = DEFAULT_GEN_MODEL
DEFAULT_INTENTS_FILE = "data/user_prompt_intents.jsonl"
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_PROJECTION_LAYER = 20


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    """Print and execute a subprocess command from the given repo directory."""
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared generation-model CLI arguments to a parser."""
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


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared selection/filtering CLI arguments to a parser."""
    parser.add_argument("--selection-mode", choices=["best_one", "top_k", "threshold"], default="best_one")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold-score", type=float, default=85.0)
    parser.add_argument("--min-neutral-score", type=int, default=75)
    parser.add_argument("--min-trait-score", type=int, default=75)
    parser.add_argument("--min-pair-score", type=int, default=80)
    parser.add_argument("--min-final-score", type=float, default=0.0)
    parser.add_argument("--no-dedupe", action="store_true")


def add_judge_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared LLM-judge CLI arguments to a parser."""
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--requests-per-second", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--neutral-weight", type=float, default=0.2)
    parser.add_argument("--trait-weight", type=float, default=0.3)
    parser.add_argument("--pair-weight", type=float, default=0.5)


def add_projection_args(
    parser: argparse.ArgumentParser,
    *,
    axes_dir_required: bool,
    include_projection_toggle: bool,
) -> None:
    """Attach shared projection CLI arguments to a parser."""
    if include_projection_toggle:
        parser.add_argument("--with-projection", action="store_true")

    parser.add_argument(
        "--axes-dir",
        type=str,
        required=axes_dir_required,
        default=None if axes_dir_required else None,
        help="Directory of .pt axis files",
    )
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


def add_plot_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared plotting CLI arguments to a parser."""
    parser.add_argument("--with-plot", action="store_true")
    parser.add_argument(
        "--plot-script",
        type=str,
        default="project/plots/plot_trait_run_axes.py",
        help="Path to plotting script",
    )
    parser.add_argument("--plot-top-k", type=int, default=20)


def build_trait_output_paths(repo_root: Path, trait: str, run_name: str) -> dict[str, Path]:
    """Return the standard output paths used by one trait/run pipeline execution."""
    trait_dir = repo_root / "outputs" / "user_prompts" / trait
    return {
        "trait_dir": trait_dir,
        "candidates_file": trait_dir / "candidates" / f"{run_name}.jsonl",
        "judged_file": trait_dir / "judged" / f"{run_name}.jsonl",
        "selected_file": trait_dir / "selected" / f"{run_name}.jsonl",
        "responses_file": trait_dir / "responses" / f"{run_name}.jsonl",
        "projections_file": trait_dir / "projections" / f"{run_name}.jsonl",
        "neutral_projections_file": trait_dir / "projections" / f"{run_name}__neutral.jsonl",
        "plot_file": trait_dir / "plots" / f"{run_name}.png",
    }


def get_projection_output_path(repo_root: Path, trait: str, run_name: str) -> Path:
    """Return the projection JSONL path for a specific trait/run pair."""
    return build_trait_output_paths(repo_root, trait, run_name)["projections_file"]
