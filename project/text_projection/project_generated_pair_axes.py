#!/usr/bin/env python3
"""
Generate two assistant responses, capture answer-time activations, and project them onto axes.

Use this script when you want to compare two prompts by the assistant's actual
internal activations while it is generating the answers, rather than by
re-encoding finished text afterwards.

For each input prompt, this script:
1. formats a one-turn user conversation
2. generates one assistant response
3. computes `answer_mean` over the generated assistant tokens
4. projects that activation onto one or more saved axes
5. saves prompts, responses, and projection scores in one JSONL file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from assistant_axis.internals import ProbingModel
    from project.pipeline_utils import load_axis_vector, resolve_axis_files
except ModuleNotFoundError:
    from assistant_axis.internals import ProbingModel
    from pipeline_utils import load_axis_vector, resolve_axis_files


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for generation-time pair projection."""
    parser = argparse.ArgumentParser(
        description="Generate two responses and project their generation-time activations onto one or more axes."
    )
    parser.add_argument("--axes-dir", type=str, required=True, help="Directory containing saved `.pt` axis files")
    parser.add_argument(
        "--projection-mode",
        choices=["all", "one", "subset"],
        default="all",
        help="Use all axes, one named axis, or a subset from a JSON list.",
    )
    parser.add_argument("--axis-trait", type=str, default=None, help="Axis trait name for --projection-mode one")
    parser.add_argument(
        "--axis-traits-file",
        type=str,
        default=None,
        help="JSON file containing axis trait names for --projection-mode subset",
    )
    parser.add_argument("--model-name", type=str, required=True, help="HF model name used for generation")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to project")
    parser.add_argument("--prompt-a", type=str, required=True, help="First user prompt")
    parser.add_argument("--prompt-b", type=str, required=True, help="Second user prompt")
    parser.add_argument("--label-a", type=str, default="prompt_a", help="Label for first prompt/response")
    parser.add_argument("--label-b", type=str, default="prompt_b", help="Label for second prompt/response")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    return parser.parse_args()


def build_chat_kwargs(model_name: str) -> dict[str, Any]:
    """Return model-specific chat-template kwargs."""
    if "qwen" in model_name.lower():
        return {"enable_thinking": False}
    return {}


def generate_response_with_answer_mean(
    pm: ProbingModel,
    conversation: list[dict[str, str]],
    *,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, torch.Tensor]:
    """Generate one response and average layer activations over generated assistant tokens."""
    tokenizer = pm.tokenizer
    model = pm.model
    device = pm.device
    chat_kwargs = build_chat_kwargs(pm.model_name)

    formatted_prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    max_prompt_tokens = max(1, max_model_len - max_new_tokens)
    prompt_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    input_ids = prompt_inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    model_layers = pm.get_layers()
    num_layers = len(model_layers)
    captured_sequence: dict[int, torch.Tensor] = {}
    handles = []

    def create_hook_fn(layer_idx: int):
        def hook_fn(module, inputs, output):
            act_tensor = output[0] if isinstance(output, tuple) else output
            captured_sequence[layer_idx] = act_tensor[0].detach().cpu()

        return hook_fn

    for layer_idx, layer_module in enumerate(model_layers):
        handles.append(layer_module.register_forward_hook(create_hook_fn(layer_idx)))

    try:
        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": prompt_inputs["attention_mask"].to(device),
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p

            output_ids = model.generate(**generation_kwargs)
            full_input_ids = output_ids[:, :max_model_len]
            _ = model(full_input_ids)
    finally:
        for handle in handles:
            handle.remove()

    generated_ids = full_input_ids[0, prompt_len:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if generated_ids.numel() == 0:
        hidden_size = model.config.hidden_size
        answer_mean = torch.zeros((num_layers, hidden_size), dtype=torch.bfloat16)
        return response_text, answer_mean

    answer_activations = []
    for layer_idx in range(num_layers):
        layer_seq = captured_sequence[layer_idx]
        if layer_seq.shape[0] != full_input_ids.shape[1]:
            raise ValueError(
                "Captured activation length does not match generated sequence length: "
                f"{layer_seq.shape[0]} vs {full_input_ids.shape[1]}"
            )
        answer_activations.append(layer_seq[prompt_len:, :].mean(dim=0))

    answer_mean = torch.stack(answer_activations).to(torch.bfloat16)
    return response_text, answer_mean


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write projection rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    """Generate both responses, project their answer_mean activations, and save the results."""
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    axis_paths = resolve_axis_files(
        repo_root=repo_root,
        axes_dir=Path(args.axes_dir),
        projection_mode=args.projection_mode,
        axis_trait=args.axis_trait,
        traits_file=args.axis_traits_file,
    )
    axes = [load_axis_vector(path, args.layer) for path in axis_paths]

    pm = ProbingModel(args.model_name)
    try:
        print(f"Generating response for {args.label_a}...", flush=True)
        response_a, act_a = generate_response_with_answer_mean(
            pm,
            [{"role": "user", "content": args.prompt_a}],
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Generating response for {args.label_b}...", flush=True)
        response_b, act_b = generate_response_with_answer_mean(
            pm,
            [{"role": "user", "content": args.prompt_b}],
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    finally:
        pm.close()

    rows: list[dict[str, Any]] = []
    for axis_info in axes:
        score_a = torch.dot(act_a[args.layer].float(), axis_info["axis"]).item()
        score_b = torch.dot(act_b[args.layer].float(), axis_info["axis"]).item()
        rows.append(
            {
                "label_a": args.label_a,
                "label_b": args.label_b,
                "prompt_a": args.prompt_a,
                "prompt_b": args.prompt_b,
                "response_a": response_a,
                "response_b": response_b,
                "projection_trait": axis_info["trait"],
                "projection_axis_path": axis_info["path"],
                "projection_activation_position": axis_info["activation_position"],
                "projection_filter_name": axis_info["filter_name"],
                "projection_layer": args.layer,
                "projection_score_a": score_a,
                "projection_score_b": score_b,
                "projection_delta_b_minus_a": score_b - score_a,
                "projection_activation_source": "answer_mean",
            }
        )

    output_path = Path(args.output)
    write_jsonl(rows, output_path)
    print(f"Saved {len(rows)} projection rows to {output_path}")


if __name__ == "__main__":
    main()
