"""
Utilities for running projection on selected prompt pairs.

This module does NOT implement projection math itself.

Instead, it:
1. Chooses which axis (.pt) files to use.
2. Iterates over selected neutral-vs-trait rows from one user-trait run.
3. Uses the default pair-projection logic in-process when possible.
4. Filters and merges results into a single JSONL output.

Key idea:
---------
Input:
- selected rows from one user-trait run
- each row has a neutral side and a trait-conditioned side
- each row is projected onto one or more assistant axes

Output:
- flattened projection rows
- one row per [selected_row x assistant_axis]

Important distinction:
- this module does not mix many user traits together
- it handles one user-trait run at a time
- the "many" dimension here is usually the assistant-axis list

Neutral side
------------
For each selected row, this module carries both sides forward:
- neutral side
- trait-conditioned side

That means the merged projection output already contains the information
needed for both:
- the main comparison file
- a reusable neutral baseline file for the same run

This module is essentially a batch runner + result merger around the
pair-projection logic.
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
    neutral_output_file: Path | None,
    projection_script: Path,
    axis_files: list[Path],
    model_name: str,
    layer: int,
    run_cmd,
) -> None:
    """
    Project each selected neutral/trait pair onto the requested axes.

    Projection is response-only. Each selected row must already contain
    `neutral_response` and `trait_response`. Runs the projection CLI once per
    selected row, then filters to the requested axes.

    When `neutral_output_file` is provided, the same merged rows are also saved
    as a dedicated neutral-baseline artifact. This keeps a reusable baseline
    file next to the main pair-projection output without re-running projection.
    """
    selected_rows = load_jsonl(selected_file)

    if not selected_rows:
        write_jsonl([], output_file)
        if neutral_output_file is not None:
            write_jsonl([], neutral_output_file)
        print(f"No selected rows found in {selected_file}; wrote empty projection file.")
        return

    if not axis_files:
        raise ValueError("No axis files provided")

    requested_traits = {axis_file.stem for axis_file in axis_files}
    axes_dir = axis_files[0].parent

    for axis_file in axis_files:
        if axis_file.parent != axes_dir:
            raise ValueError("All axis files must come from the same axes directory")

    use_optimized_path = projection_script.name == "project_pair_axes.py"

    all_rows: list[dict[str, Any]] = []
    if use_optimized_path:
        from project_pair_axes import load_axis_vector, project_text_pair
        from project_text_axis import load_model

        axes = [load_axis_vector(path, layer) for path in axis_files]

        print(
            f"Using in-process projection path for {len(selected_rows)} rows "
            f"across {len(axes)} axes."
        )
        model, tokenizer = load_model(model_name)

        for idx, row in enumerate(selected_rows):
            if not row.get("neutral_response") or not row.get("trait_response"):
                raise ValueError(
                    "Projection requires response fields on every selected row. "
                    f"Missing response text in row index {idx} from {selected_file}. "
                    "Run response generation first instead of projecting prompts."
                )

            label_a = "neutral_response"
            label_b = "trait_response"
            pair_rows = project_text_pair(
                model=model,
                tokenizer=tokenizer,
                axes=axes,
                text_a=row["neutral_response"],
                text_b=row["trait_response"],
                label_a=label_a,
                label_b=label_b,
                layer=layer,
            )

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
                        "projection_text_source_neutral": label_a,
                        "projection_text_source_trait": label_b,
                    }
                )
                all_rows.append(merged)
    else:
        temp_dir = output_file.parent / ".tmp_pair_projection"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(selected_rows):
            if not row.get("neutral_response") or not row.get("trait_response"):
                raise ValueError(
                    "Projection requires response fields on every selected row. "
                    f"Missing response text in row index {idx} from {selected_file}. "
                    "Run response generation first instead of projecting prompts."
                )

            pair_out = temp_dir / f"pair_{idx:05d}.jsonl"
            text_a = row["neutral_response"]
            text_b = row["trait_response"]
            label_a = "neutral_response"
            label_b = "trait_response"

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
                text_a,
                "--text-b",
                text_b,
                "--label-a",
                label_a,
                "--label-b",
                label_b,
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
                        "projection_text_source_neutral": label_a,
                        "projection_text_source_trait": label_b,
                    }
                )
                all_rows.append(merged)

    write_jsonl(all_rows, output_file)
    print(f"Saved {len(all_rows)} projection rows to {output_file}")
    if neutral_output_file is not None:
        write_jsonl(all_rows, neutral_output_file)
        print(f"Saved {len(all_rows)} neutral baseline rows to {neutral_output_file}")
