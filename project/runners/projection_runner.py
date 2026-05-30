"""
Projects one trait run's hidden-state activations onto personality axes.

The core idea
-------------
During stage 4 (response generation), the model's internal activations are
captured while it generates each response — specifically the mean residual-stream
hidden state over the answer tokens at a chosen layer (answer_mean). These
activations encode how the model internally represented its response.

Each personality axis is a direction vector in that same activation space,
precomputed by contrasting responses at opposite poles of a trait (e.g.
"very organised" vs "very disorganised"). Projecting an activation onto an
axis is a dot product: a high score means the model's internal state strongly
aligns with that personality direction; a low/negative score means the opposite.

What this module does
---------------------
For each selected example × each personality axis it computes:

  projection_score_neutral = dot(neutral_answer_mean[layer], axis_vector)
  projection_score_trait   = dot(trait_answer_mean[layer],   axis_vector)
  projection_delta         = projection_score_trait - projection_score_neutral

The delta is the key number: how much does the user trait shift the model's
responses along this personality axis compared to a neutral user?

Inputs  → responses JSONL (stage 4) + activations .pt (stage 4) + axis .pt files
Output  → projections JSONL with N_examples × N_axes rows, one per combination
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

    `neutral_output_file` is accepted for backwards compatibility but ignored —
    every output row already contains `projection_score_neutral` so a separate
    file would be a duplicate.
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
