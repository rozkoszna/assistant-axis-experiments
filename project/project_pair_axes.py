#!/usr/bin/env python3
"""
Project a pair of texts onto one or more trait axes.

This script:
1. Resolves which saved axis files to use
2. Computes one vector for text A
3. Computes one vector for text B
4. Projects both onto each requested trait axis
5. Saves per-trait scores and differences

Notes:
- This is best aligned with axes built using:
      activation_position = "answer_mean"
- It uses mean pooling over all tokens in each input text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from project_text_axis import load_model, text_vector, project
from projection_runner import resolve_axis_files


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for projecting two texts onto one or more axes."""
    parser = argparse.ArgumentParser(
        description="Project a pair of texts onto one or more trait axes."
    )
    parser.add_argument(
        "--axes-dir",
        type=str,
        required=True,
        help="Directory containing saved `.pt` trait axis files",
    )
    parser.add_argument(
        "--projection-mode",
        choices=["all", "one", "subset"],
        default="all",
        help="Use all axes, one named axis, or a subset from a JSON list.",
    )
    parser.add_argument(
        "--axis-trait",
        type=str,
        default=None,
        help="Axis trait name for --projection-mode one",
    )
    parser.add_argument(
        "--axis-traits-file",
        type=str,
        default=None,
        help="JSON file containing axis trait names for --projection-mode subset",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name used for projection",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to use from each saved vector stack",
    )
    parser.add_argument(
        "--text-a",
        type=str,
        required=True,
        help="First text to score",
    )
    parser.add_argument(
        "--text-b",
        type=str,
        required=True,
        help="Second text to score",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="text_a",
        help="Label for first text in the output",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="text_b",
        help="Label for second text in the output",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path",
    )
    return parser.parse_args()


def load_axis_vector(path: Path, layer: int) -> dict[str, Any]:
    """Load one normalized axis vector and its metadata from a `.pt` file."""
    data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected axis file format in {path}: {type(data)}")

    if "vector" not in data:
        raise KeyError(f"No 'vector' key found in {path}. Available keys: {list(data.keys())}")

    vectors = data["vector"].float().cpu()

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 'vector' in {path} to have shape [n_layers, d_model], "
            f"got {tuple(vectors.shape)}"
        )

    if not (0 <= layer < vectors.shape[0]):
        raise ValueError(
            f"Layer {layer} out of range for {path}; "
            f"vector stack shape is {tuple(vectors.shape)}"
        )

    axis = vectors[layer]
    axis = axis / axis.norm()

    return {
        "trait": data.get("trait", path.stem),
        "activation_position": data.get("activation_position"),
        "filter_name": data.get("filter_name"),
        "axis": axis,
        "path": str(path),
    }


def iter_axis_files(axes_dir: Path):
    """Yield all saved axis files in a directory in sorted order."""
    for path in sorted(axes_dir.glob("*.pt")):
        yield path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write projection rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_axes_for_projection(
    *,
    repo_root: Path,
    axes_dir: Path,
    projection_mode: str,
    axis_trait: str | None,
    axis_traits_file: str | None,
    layer: int,
) -> list[dict[str, Any]]:
    """Resolve and load all requested axis vectors for one projection job."""
    axis_paths = resolve_axis_files(
        repo_root=repo_root,
        axes_dir=axes_dir,
        projection_mode=projection_mode,
        axis_trait=axis_trait,
        traits_file=axis_traits_file,
    )
    if not axis_paths:
        raise ValueError(f"No .pt files found in {axes_dir}")
    return [load_axis_vector(path, layer) for path in axis_paths]


def project_text_pair(
    *,
    model,
    tokenizer,
    axes: list[dict[str, Any]],
    text_a: str,
    text_b: str,
    label_a: str,
    label_b: str,
    layer: int,
) -> list[dict[str, Any]]:
    """Project one text pair onto a preloaded list of axes using one loaded model."""
    vec_a = text_vector(model, tokenizer, text_a, layer)
    vec_b = text_vector(model, tokenizer, text_b, layer)

    results: list[dict[str, Any]] = []
    for axis_info in axes:
        score_a = project(vec_a, axis_info["axis"])
        score_b = project(vec_b, axis_info["axis"])
        delta_b_minus_a = score_b - score_a

        results.append(
            {
                "trait": axis_info["trait"],
                "axis_path": axis_info["path"],
                "activation_position": axis_info["activation_position"],
                "filter_name": axis_info["filter_name"],
                "layer": layer,
                "label_a": label_a,
                "label_b": label_b,
                "text_a": text_a,
                "text_b": text_b,
                "score_a": score_a,
                "score_b": score_b,
                "delta_a_minus_b": score_a - score_b,
                "delta_b_minus_a": delta_b_minus_a,
            }
        )

    results.sort(key=lambda x: abs(x["delta_b_minus_a"]), reverse=True)
    return results


def main() -> None:
    """Project both input texts onto the requested axes and save all per-axis scores."""
    args = parse_args()

    axes_dir = Path(args.axes_dir)
    repo_root = Path(__file__).resolve().parent.parent
    axes = load_axes_for_projection(
        repo_root=repo_root,
        axes_dir=axes_dir,
        projection_mode=args.projection_mode,
        axis_trait=args.axis_trait,
        axis_traits_file=args.axis_traits_file,
        layer=args.layer,
    )
    print(f"Found {len(axes)} axis files for projection")

    model, tokenizer = load_model(args.model_name)
    results = project_text_pair(
        model=model,
        tokenizer=tokenizer,
        axes=axes,
        text_a=args.text_a,
        text_b=args.text_b,
        label_a=args.label_a,
        label_b=args.label_b,
        layer=args.layer,
    )
    
    output_path = Path(args.output)
    write_jsonl(results, output_path)
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
