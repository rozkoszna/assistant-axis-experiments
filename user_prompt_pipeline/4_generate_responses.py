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
from assistant_axis.internals import ProbingModel  # noqa: E402


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


def save_activation_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    parser.add_argument("--activations-file", type=str, default=None)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample one token id from logits using greedy or top-p sampling."""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = True

        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_indices[keep_mask]] = probs[sorted_indices[keep_mask]]
        probs = filtered_probs / filtered_probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


def build_chat_kwargs(model_name: str) -> dict[str, Any]:
    """Return chat-template kwargs for model-specific quirks."""
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
    """Generate one assistant response and capture generation-time answer_mean activations."""
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
    attention_mask = prompt_inputs["attention_mask"].to(device)

    model_layers = pm.get_layers()
    num_layers = len(model_layers)
    captured_last_token: dict[int, torch.Tensor] = {}
    handles = []

    def create_hook_fn(layer_idx: int):
        def hook_fn(module, inputs, output):
            act_tensor = output[0] if isinstance(output, tuple) else output
            captured_last_token[layer_idx] = act_tensor[0, -1, :].detach().cpu()

        return hook_fn

    for layer_idx, layer_module in enumerate(model_layers):
        handles.append(layer_module.register_forward_hook(create_hook_fn(layer_idx)))

    generated_token_ids: list[int] = []
    generated_activations: list[torch.Tensor] = []
    eos_token_id = tokenizer.eos_token_id

    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = sample_next_token(outputs.logits[0, -1, :], temperature, top_p)

            for _ in range(max_new_tokens):
                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                generated_token_ids.append(next_token_id)

                step_input_ids = torch.tensor([[next_token_id]], device=device)
                step_attention_mask = torch.ones_like(step_input_ids, device=device)
                captured_last_token.clear()

                outputs = model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                token_activation = torch.stack(
                    [captured_last_token[layer_idx] for layer_idx in range(num_layers)]
                )
                generated_activations.append(token_activation)
                next_token_id = sample_next_token(outputs.logits[0, -1, :], temperature, top_p)
    finally:
        for handle in handles:
            handle.remove()

    response_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    if generated_activations:
        answer_mean = torch.stack(generated_activations).mean(dim=0).to(torch.bfloat16)
    else:
        hidden_size = model.config.hidden_size
        answer_mean = torch.zeros((num_layers, hidden_size), dtype=torch.bfloat16)

    return response_text, answer_mean


def main() -> None:
    args = parse_args()
    selected_file = Path(args.selected_file)
    output_file = Path(args.output_file)
    activations_file = Path(args.activations_file) if args.activations_file else None

    rows = load_selected(selected_file)
    if not rows:
        save_rows(output_file, [])
        if activations_file is not None:
            save_activation_payload(
                activations_file,
                {
                    "activation_position": "answer_mean",
                    "model_name": args.model,
                    "rows": [],
                },
            )
        print(f"No selected rows found in {selected_file}; wrote empty responses file.")
        return

    if activations_file is None:
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
        return

    pm = ProbingModel(args.model)
    enriched_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []

    try:
        for row in rows:
            neutral_conversation = [{"role": "user", "content": row["neutral_prompt"]}]
            trait_conversation = [{"role": "user", "content": row["trait_prompt"]}]

            neutral_response, neutral_answer_mean = generate_response_with_answer_mean(
                pm,
                neutral_conversation,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            trait_response, trait_answer_mean = generate_response_with_answer_mean(
                pm,
                trait_conversation,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            enriched = dict(row)
            enriched["neutral_response"] = neutral_response
            enriched["trait_response"] = trait_response
            enriched_rows.append(enriched)

            activation_rows.append(
                {
                    "neutral_answer_mean": neutral_answer_mean,
                    "trait_answer_mean": trait_answer_mean,
                }
            )
    finally:
        pm.close()

    save_rows(output_file, enriched_rows)
    save_activation_payload(
        activations_file,
        {
            "activation_position": "answer_mean",
            "model_name": args.model,
            "rows": activation_rows,
        },
    )
    print(f"Saved {len(enriched_rows)} response rows to {output_file}")
    print(f"Saved {len(activation_rows)} activation rows to {activations_file}")


if __name__ == "__main__":
    main()
