#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from io_utils import load_jsonl, print_header, print_kv, write_json, write_jsonl
from pipeline_utils import run_cmd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the user-trait pipeline multiple times and measure projection variation."
    )

    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--run-prefix", type=str, default=None)

    parser.add_argument("--axis", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--projection-model", type=str, default=DEFAULT_MODEL)

    parser.add_argument(
        "--text-keys",
        nargs="+",
        default=["trait_response", "response", "text", "prompt", "user_prompt"],
        help="Candidate keys to use when extracting prompt text from selected rows.",
    )

    parser.add_argument(
        "--hidden-state-indexing",
        choices=["as_is", "layer_plus_one"],
        default="as_is",
        help="Use hidden_states[layer] or hidden_states[layer + 1].",
    )

    parser.add_argument("--output-dir", type=str, required=True)

    return parser.parse_args()
def load_axis(axis_path: Path, layer: int) -> dict[str, Any]:
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


def load_model(model_name: str):
    print(f"Loading model: {model_name}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def text_vector(
    model,
    tokenizer,
    text: str,
    layer: int,
    hidden_state_indexing: str,
) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if hidden_state_indexing == "as_is":
        hidden_idx = layer
    elif hidden_state_indexing == "layer_plus_one":
        hidden_idx = layer + 1
    else:
        raise ValueError(f"Unknown hidden_state_indexing: {hidden_state_indexing}")

    if hidden_idx >= len(out.hidden_states):
        raise ValueError(
            f"Requested hidden state index {hidden_idx}, "
            f"but model returned only {len(out.hidden_states)} hidden states"
        )

    hidden = out.hidden_states[hidden_idx]
    return hidden.mean(dim=1).squeeze(0).float().cpu()


def project(vec: torch.Tensor, axis: torch.Tensor) -> float:
    axis = axis / axis.norm()
    return torch.dot(vec, axis).item()


def summarize(scores: list[float]) -> dict[str, Any]:
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


def extract_text(row: dict[str, Any], text_keys: list[str]) -> tuple[str, str]:
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value, key

    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return value, key

    raise KeyError(f"Could not find text field in row. Available keys: {list(row.keys())}")


def build_pipeline_cmd(trait: str, run_name: str, seed: int) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "user_prompt_pipeline/run_user_trait_pipeline.py",
        "--trait",
        trait,
        "--run-name",
        run_name,
        "--seed",
        str(seed),
    ]


def build_selected_path(trait: str, run_name: str) -> Path:
    return REPO_ROOT / "outputs" / "user_prompts" / trait / "selected" / f"{run_name}.jsonl"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    axis_info = load_axis(Path(args.axis), args.layer)
    axis = axis_info["axis"]

    print_header("Trait Axis Variation")
    print_kv("Trait", args.trait)
    print_kv("Axis", args.axis)
    print_kv("Layer", args.layer)
    print_kv("Projection model", args.projection_model)
    print_kv("Runs", args.num_runs)
    print_kv("Base seed", args.base_seed)
    print_kv("Output dir", out_dir)

    model, tokenizer = load_model(args.projection_model)

    run_prefix = args.run_prefix or f"{args.trait}_variation"

    all_rows: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []

    for i in range(args.num_runs):
        seed = args.base_seed + i
        run_name = f"{run_prefix}_{i:03d}_seed_{seed}"

        run_cmd(build_pipeline_cmd(args.trait, run_name, seed), cwd=REPO_ROOT)

        selected_path = build_selected_path(args.trait, run_name)
        if not selected_path.exists():
            raise FileNotFoundError(f"Selected file not found: {selected_path}")

        selected_rows = load_jsonl(selected_path)
        run_scores: list[float] = []

        for selected_idx, row in enumerate(selected_rows):
            text, text_key = extract_text(row, args.text_keys)

            vec = text_vector(
                model=model,
                tokenizer=tokenizer,
                text=text,
                layer=args.layer,
                hidden_state_indexing=args.hidden_state_indexing,
            )
            score = project(vec, axis)
            run_scores.append(score)

            all_rows.append(
                {
                    "trait": args.trait,
                    "run_name": run_name,
                    "run_idx": i,
                    "seed": seed,
                    "selected_idx": selected_idx,
                    "axis_path": args.axis,
                    "axis_trait": axis_info["trait"],
                    "activation_position": axis_info["activation_position"],
                    "projection_model": args.projection_model,
                    "layer": args.layer,
                    "text_key_used": text_key,
                    "text": text,
                    "projection_score": score,
                    "selected_row": row,
                }
            )

        run_summary = summarize(run_scores)
        run_summary.update(
            {
                "trait": args.trait,
                "run_name": run_name,
                "run_idx": i,
                "seed": seed,
                "num_selected": len(selected_rows),
                "axis_path": args.axis,
                "axis_trait": axis_info["trait"],
            }
        )
        per_run_rows.append(run_summary)

        mean_str = "None" if run_summary["mean"] is None else f"{run_summary['mean']:.4f}"
        std_str = "None" if run_summary["std"] is None else f"{run_summary['std']:.4f}"
        print(f"{run_name}: mean={mean_str} std={std_str} n={run_summary['count']}")

    overall_scores = [row["projection_score"] for row in all_rows]
    overall_summary = summarize(overall_scores)
    overall_summary.update(
        {
            "trait": args.trait,
            "axis_path": args.axis,
            "axis_trait": axis_info["trait"],
            "activation_position": axis_info["activation_position"],
            "projection_model": args.projection_model,
            "layer": args.layer,
            "num_runs_requested": args.num_runs,
            "num_runs_completed": len(per_run_rows),
            "run_prefix": run_prefix,
            "base_seed": args.base_seed,
            "hidden_state_indexing": args.hidden_state_indexing,
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
