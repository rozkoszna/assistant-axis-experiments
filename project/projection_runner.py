"""
Utilities for running projection on selected prompt pairs.

This module does NOT implement projection math itself.

Instead, it:
1. Chooses which axis (.pt) files to use.
2. Iterates over selected prompt pairs (neutral vs trait prompts).
3. Calls the projection script (e.g. project_pair_axes.py) once per pair.
4. Filters and merges results into a single JSONL output.

Key idea:
---------
Input:  selected rows (each containing a neutral_prompt and trait_prompt)
Output: flattened projection rows (one row per [selected_row × axis])

This module is essentially a batch runner + result merger around the
projection CLI script.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

from io_utils import load_jsonl, load_trait_list, write_jsonl


def resolve_axis_files(
    repo_root: Path,
    axes_dir: Path,
    projection_mode: str,
    axis_trait: str | None,
    traits_file: str | None,
) -> list[Path]:
    """
    Resolve which axis .pt files should be used for projection.

    Modes:
    - all:    every .pt file in axes_dir
    - one:    exactly one axis specified by --axis-trait
    - subset: a JSON file containing a list of axis trait names
    """
    if not axes_dir.exists():
        raise FileNotFoundError(f"Axes directory not found: {axes_dir}")

    if projection_mode == "all":
        axis_files = sorted(axes_dir.glob("*.pt"))
        if not axis_files:
            raise ValueError(f"No .pt files found in {axes_dir}")
        return axis_files

    if projection_mode == "one":
        if not axis_trait:
            raise ValueError("--projection-mode one requires --axis-trait")

        axis_file = axes_dir / f"{axis_trait}.pt"
        if not axis_file.exists():
            raise FileNotFoundError(f"Axis file not found: {axis_file}")
        return [axis_file]

    if projection_mode == "subset":
        if not traits_file:
            raise ValueError("--projection-mode subset requires --axis-traits-file")

        trait_names = load_trait_list(repo_root / traits_file)
        axis_files: list[Path] = []
        missing: list[str] = []

        for trait_name in trait_names:
            axis_file = axes_dir / f"{trait_name}.pt"
            if axis_file.exists():
                axis_files.append(axis_file)
            else:
                missing.append(trait_name)

        if missing:
            raise FileNotFoundError(
                f"Missing axis files for traits: {missing} in {axes_dir}"
            )

        if not axis_files:
            raise ValueError(f"No axis files resolved from {traits_file}")

        return axis_files

    raise ValueError(f"Unknown projection mode: {projection_mode}")


def run_projection_for_selected(
    *,
    selected_file: Path,
    output_file: Path,
    projection_script: Path,
    axis_files: list[Path],
    model_name: str,
    layer: int,
    run_cmd,
) -> None:
    """
    Project each selected neutral/trait prompt pair onto the requested axes.

    Runs the projection CLI once per selected row, then filters to the requested axes.
    """
    selected_rows = load_jsonl(selected_file)

    if not selected_rows:
        write_jsonl([], output_file)
        print(f"No selected rows found in {selected_file}; wrote empty projection file.")
        return

    if not axis_files:
        raise ValueError("No axis files provided")

    requested_traits = {axis_file.stem for axis_file in axis_files}
    axes_dir = axis_files[0].parent

    for axis_file in axis_files:
        if axis_file.parent != axes_dir:
            raise ValueError("All axis files must come from the same axes directory")

    all_rows: list[dict[str, Any]] = []
    temp_dir = output_file.parent / ".tmp_pair_projection"
    temp_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(selected_rows):
        pair_out = temp_dir / f"pair_{idx:05d}.jsonl"

        cmd = [
            "uv",
            "run",
            "python",
            str(projection_script),
            "--axes-dir",
            str(axes_dir),
            "--model-name",
            model_name,
            "--layer",
            str(layer),
            "--text-a",
            row["neutral_prompt"],
            "--text-b",
            row["trait_prompt"],
            "--label-a",
            "neutral_prompt",
            "--label-b",
            "trait_prompt",
            "--output",
            str(pair_out),
        ]
        run_cmd(cmd)

        pair_rows = load_jsonl(pair_out)
        pair_rows = [r for r in pair_rows if r["trait"] in requested_traits]

        for pair_row in pair_rows:
            merged = dict(row)
            merged.update(
                {
                    "projection_trait": pair_row["trait"],
                    "projection_axis_path": pair_row["axis_path"],
                    "projection_activation_position": pair_row["activation_position"],
                    "projection_filter_name": pair_row["filter_name"],
                    "projection_layer": pair_row["layer"],
                    "projection_score_neutral": pair_row["score_a"],
                    "projection_score_trait": pair_row["score_b"],
                    "projection_delta_trait_minus_neutral": pair_row["delta_b_minus_a"],
                }
            )
            all_rows.append(merged)

    write_jsonl(all_rows, output_file)
    print(f"Saved {len(all_rows)} projection rows to {output_file}")