#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import jsonlines
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.generation import VLLMGenerator  # noqa: E402


def load_selected(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonlines.open(path, "r") as reader:
        for row in reader:
            rows.append(dict(row))
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def compute_tensor_parallel_size(user_value: int | None) -> int:
    if user_value is not None:
        return user_value

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = [x.strip() for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]
        return max(1, len(visible))

    count = torch.cuda.device_count()
    return max(1, count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate assistant responses for selected user prompt pairs.")
    parser.add_argument("--selected-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_file = Path(args.selected_file)
    output_file = Path(args.output_file)

    rows = load_selected(selected_file)
    if not rows:
        save_rows(output_file, [])
        print(f"No selected rows found in {selected_file}; wrote empty responses file.")
        return

    tensor_parallel_size = compute_tensor_parallel_size(args.tensor_parallel_size)
    generator = VLLMGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    generator.load()

    conversations: list[list[dict[str, str]]] = []
    for row in rows:
        conversations.append([{"role": "user", "content": row["neutral_prompt"]}])
        conversations.append([{"role": "user", "content": row["trait_prompt"]}])

    outputs = generator.generate_batch(conversations)
    if len(outputs) != 2 * len(rows):
        raise RuntimeError(f"Expected {2 * len(rows)} responses, got {len(outputs)}")

    enriched_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        enriched = dict(row)
        enriched["neutral_response"] = outputs[2 * idx]
        enriched["trait_response"] = outputs[2 * idx + 1]
        enriched_rows.append(enriched)

    save_rows(output_file, enriched_rows)
    print(f"Saved {len(enriched_rows)} response rows to {output_file}")


if __name__ == "__main__":
    main()
