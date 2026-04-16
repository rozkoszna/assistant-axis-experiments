#!/usr/bin/env python3
"""
Project a pair of texts onto all trait axes in a directory.

This script:
1. Loads all trait axis files in a directory
2. Computes one vector for text A
3. Computes one vector for text B
4. Projects both onto every trait axis
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

from project_prompt_axis import load_model, text_vector, project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a pair of texts onto all trait axes in a directory."
    )
    parser.add_argument(
        "--axes-dir",
        type=str,
        required=True,
        help="Directory containing many .pt trait axis files",
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
    for path in sorted(axes_dir.glob("*.pt")):
        yield path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()

    axes_dir = Path(args.axes_dir)
    axis_paths = list(iter_axis_files(axes_dir))
    if not axis_paths:
        raise ValueError(f"No .pt files found in {axes_dir}")

    print(f"Found {len(axis_paths)} axis files in {axes_dir}")
    axes = [load_axis_vector(path, args.layer) for path in axis_paths]

    model, tokenizer = load_model(args.model_name)

    vec_a = text_vector(model, tokenizer, args.text_a, args.layer)
    vec_b = text_vector(model, tokenizer, args.text_b, args.layer)

    results: list[dict[str, Any]] = []
    for axis_info in axes:
        score_a = project(vec_a, axis_info["axis"])
        score_b = project(vec_b, axis_info["axis"])
        delta = score_a - score_b

        results.append(
            {
                "trait": axis_info["trait"],
                "axis_path": axis_info["path"],
                "activation_position": axis_info["activation_position"],
                "filter_name": axis_info["filter_name"],
                "layer": args.layer,
                "label_a": args.label_a,
                "label_b": args.label_b,
                "text_a": args.text_a,
                "text_b": args.text_b,
                "score_a": score_a,
                "score_b": score_b,
                "delta_a_minus_b": delta,
            }
        )

    results.sort(key=lambda x: abs(x["delta_a_minus_b"]), reverse=True)

    output_path = Path(args.output)
    write_jsonl(results, output_path)
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()