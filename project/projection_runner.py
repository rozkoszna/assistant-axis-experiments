"""
Helpers for projecting one user-trait run onto assistant axes.

What this module does
---------------------
This module sits between the saved pipeline artifacts and the final projection
JSONL file.

It does not define the axis math itself. Instead, it:
- chooses which saved axis files to use
- reads one run's saved rows
- projects the neutral and trait sides onto the requested assistant axes
- writes a flattened projection file

What goes in
------------
This module handles one user-trait run at a time.

Its inputs are typically:
- a selected/responses file for one run
- optionally, a saved activation file for that same run
- one or more assistant axis files

Each row is still a matched pair:
- neutral side
- trait-conditioned side

What it prefers
---------------
Best case:
- use saved generation-time activations from `activations/<run_name>.pt`

Older fallback:
- re-encode the saved response text if no activation file exists

It should never project prompts.

What comes out
--------------
The output is a flattened JSONL file with one row per:
- [selected example x assistant axis]

So one run can produce many projection rows because:
- the user trait stays fixed
- the assistant axis changes row by row

Neutral side
------------
The neutral side is the matched baseline from the same run.

That is why this module can also write:
- `projections/<run_name>__neutral.jsonl`

This is not a separate neutral experiment. It is the neutral view of the same
run across the same assistant axes.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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
    activations_file: Path | None,
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

    Projection is response-only. New pipeline runs should provide a saved
    activation tensor file extracted during response generation. When that file
    is present, projection uses those assistant-response activations directly.

    Older runs may omit the activation file; in that case this helper falls
    back to the existing text-based pair projection logic.

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

    all_rows: list[dict[str, Any]] = []
    if activations_file is not None and activations_file.exists():
        from project_pair_axes import load_axis_vector

        activation_payload = torch.load(activations_file, map_location="cpu", weights_only=False)
        activation_rows = activation_payload.get("rows")
        if activation_rows is None:
            raise ValueError(f"Expected 'rows' in activation file: {activations_file}")
        if len(activation_rows) != len(selected_rows):
            raise ValueError(
                f"Activation row count {len(activation_rows)} does not match selected row count "
                f"{len(selected_rows)} for {activations_file}"
            )

        axes = [load_axis_vector(path, layer) for path in axis_files]
        print(
            f"Using saved generation-time activations for {len(selected_rows)} rows "
            f"across {len(axes)} axes."
        )

        for idx, (row, activation_row) in enumerate(zip(selected_rows, activation_rows)):
            neutral_act = activation_row.get("neutral_answer_mean")
            trait_act = activation_row.get("trait_answer_mean")
            if neutral_act is None or trait_act is None:
                raise ValueError(
                    f"Missing saved answer_mean activations for row index {idx} in {activations_file}"
                )

            neutral_act = neutral_act.float().cpu()
            trait_act = trait_act.float().cpu()

            for axis_info in axes:
                score_a = torch.dot(neutral_act[layer], axis_info["axis"]).item()
                score_b = torch.dot(trait_act[layer], axis_info["axis"]).item()

                merged = dict(row)
                merged.update(
                    {
                        "projection_trait": axis_info["trait"],
                        "projection_axis_path": axis_info["path"],
                        "projection_activation_position": axis_info["activation_position"],
                        "projection_filter_name": axis_info["filter_name"],
                        "projection_layer": layer,
                        "projection_score_neutral": score_a,
                        "projection_score_trait": score_b,
                        "projection_delta_trait_minus_neutral": score_b - score_a,
                        "projection_text_source_neutral": "neutral_response",
                        "projection_text_source_trait": "trait_response",
                        "projection_activation_source": activation_payload.get(
                            "activation_position", "answer_mean"
                        ),
                    }
                )
                all_rows.append(merged)
    else:
        use_optimized_path = projection_script.name == "project_pair_axes.py"
        if use_optimized_path:
            from project_pair_axes import load_axis_vector, project_text_pair
            from project_text_axis import load_model

            axes = [load_axis_vector(path, layer) for path in axis_files]

            print(
                f"Using in-process text projection path for {len(selected_rows)} rows "
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
                            "projection_activation_source": "reencoded_response_text",
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
                            "projection_activation_source": "custom_projection_script",
                        }
                    )
                    all_rows.append(merged)

    write_jsonl(all_rows, output_file)
    print(f"Saved {len(all_rows)} projection rows to {output_file}")
    if neutral_output_file is not None:
        write_jsonl(all_rows, neutral_output_file)
        print(f"Saved {len(all_rows)} neutral baseline rows to {neutral_output_file}")
