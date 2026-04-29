"""
Helpers for projecting one user-trait run onto assistant axes.

What this module does
---------------------
This module sits between the saved pipeline artifacts and the final projection
JSONL file.

It does not define the axis math itself. Instead, it:
- chooses which saved axis files to use
- reads one run's saved rows
- projects generation-time internal activations for the neutral and trait
  sides onto the requested assistant axes
- writes a flattened projection file

What goes in
------------
This module handles one user-trait run at a time.

Its inputs are typically:
- a selected/responses file for one run
- optionally, a saved activation file for that same run
- one or more assistant axis files

Important: projection uses model internals, not raw text.
The projected vectors are hidden-state activations (`answer_mean`) captured
during assistant response generation.

Each row is still a matched pair:
- neutral side
- trait-conditioned side

What it requires
----------------
Projection now expects saved generation-time activations from:
- `activations/<run_name>.pt`

If that activation file is missing, it should error.

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

from io_utils import load_jsonl, write_jsonl
from pipeline_utils import load_axis_vector, resolve_axis_files


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

    Projection is response-only and activation-based. The run must provide a
    saved activation tensor file extracted during response generation.

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

    if activations_file is None or not activations_file.exists():
        raise FileNotFoundError(
            "Projection now requires a saved activation file from response generation. "
            f"Expected: {activations_file}"
        )

    all_rows: list[dict[str, Any]] = []
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

    write_jsonl(all_rows, output_file)
    print(f"Saved {len(all_rows)} projection rows to {output_file}")
    if neutral_output_file is not None:
        write_jsonl(all_rows, neutral_output_file)
        print(f"Saved {len(all_rows)} neutral baseline rows to {neutral_output_file}")
