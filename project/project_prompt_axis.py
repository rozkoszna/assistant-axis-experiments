"""
Project text or model responses onto a precomputed axis.

This script:
1. Loads a saved axis (e.g. assistiveness, confusion)
2. Converts text into a hidden-state vector using the model
3. Projects that vector onto the axis (dot product)
4. Outputs a scalar score per input

----------------------------------------
USAGE EXAMPLES

1) Project a single text:

uv run python project/project_prompt_axis.py \
  --axis precomputed_axis/answer_mean/filter_matched_pairs_ge_50_count_ge_10_total/supportive.pt \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layer 20 \
  --text "I'm really confused about recursion. Can you explain it step by step?"

----------------------------------------

2) Project many texts from a JSONL file:

Input format (JSONL):
{"id": "1", "text": "Explain recursion simply."}
{"id": "2", "text": "I'm really confused about recursion."}

Run:

uv run python project/project_prompt_axis.py \
  --axis precomputed_axis/answer_mean/filter_matched_pairs_ge_50_count_ge_10_total/supportive.pt \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --layer 20 \
  --input-jsonl data/texts.jsonl \
  --text-key text \
  --output outputs/projected_texts.jsonl

----------------------------------------

OUTPUT

Each input gets a new field:
"projection_score" = dot(text_vector, axis)

Higher score → more aligned with the axis direction
Lower score → less aligned

----------------------------------------

AXIS FILE FORMAT

- Each .pt file contains:
    {
        "vector": Tensor of shape [n_layers, d_model],
        "trait": str,
        "activation_position": str,
    }


- The "vector" tensor contains one axis per layer.
- The script selects a layer using --layer.
- The axis used for projection is: vector[layer].
----------------------------------------

REQUIREMENTS

- You must specify:
    --model-name : HuggingFace model used to compute hidden states
    --layer      : which layer to use from the axis file

- The model must match the one used to generate the trait vectors.

NOTES

- This script uses mean pooling over all tokens in the input text.

- It is best aligned with axes built using:
      activation_position = "answer_mean"

- It does NOT faithfully reproduce:
      pre_generation_last_token
      assistant_header_mean

  Using those axes with this script may produce inconsistent results.

- Different layers may produce very different projection scores.
  It is often useful to compare multiple layers (e.g., 8, 16, 24, 31).

----------------------------------------

INTERPRETATION

- Projection scores are relative, not absolute.
- Compare scores across inputs for the same axis and layer.
- Higher magnitude = stronger alignment with the trait direction.

----------------------------------------
"""


import argparse

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Project text onto a saved axis.")
    parser.add_argument("--axis", type=str, required=True, help="Path to saved axis .pt file")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to use from the saved vector stack")
    parser.add_argument("--model-name", type=str, required=True, help="HF model name used for projection")
    parser.add_argument("--text", type=str, default=None, help="Text to project directly")
    parser.add_argument("--input-jsonl", type=str, default=None, help="Optional JSONL file with texts")
    parser.add_argument("--text-key", type=str, default="text", help="Key to read from JSONL rows")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSONL path")
    return parser.parse_args()



def load_axis(axis_path: Path, layer: int):
    data = torch.load(axis_path, map_location="cpu")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected axis file format: {type(data)}")

    if "vector" not in data:
        raise KeyError(f"No 'vector' key found. Available keys: {list(data.keys())}")

    vectors = data["vector"].float().cpu()

    if vectors.ndim != 2:
        raise ValueError(f"Expected 'vector' to have shape [n_layers, d_model], got {tuple(vectors.shape)}")

    if not (0 <= layer < vectors.shape[0]):
        raise ValueError(f"Layer {layer} out of range for vector stack of shape {tuple(vectors.shape)}")

    axis = vectors[layer]
    activation_position = data.get("activation_position")
    trait = data.get("trait")

    return axis, activation_position, trait


def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def text_vector(model, tokenizer, text: str, layer: int) -> torch.Tensor:
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

    hidden = out.hidden_states[layer]
    vec = hidden.mean(dim=1).squeeze(0).float().cpu()
    return vec


def project(vec: torch.Tensor, axis: torch.Tensor) -> float:
    axis = axis / axis.norm()
    return torch.dot(vec, axis).item()


def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    args = parse_args()

    axis_path = Path(args.axis)
    axis, activation_position, trait = load_axis(axis_path, args.layer)
    model, tokenizer = load_model(args.model_name)
    layer = args.layer

    if args.text is not None:
        vec = text_vector(model, tokenizer, args.text, layer)
        score = project(vec, axis)
        print(json.dumps({
            "text": args.text,
            "score": score,
            "axis": str(axis_path),
            "activation_position": activation_position,
            "model_name": args.model_name,
            "layer": layer,
        }, indent=2))
        return

    if args.input_jsonl is not None:
        rows_out = []
        for row in read_jsonl(Path(args.input_jsonl)):
            text = row[args.text_key]
            vec = text_vector(model, tokenizer, text, layer)
            score = project(vec, axis)

            new_row = dict(row)
            new_row["projection_score"] = score
            rows_out.append(new_row)

        if args.output is None:
            raise ValueError("When using --input-jsonl, you must also provide --output")

        write_jsonl(rows_out, Path(args.output))
        print(f"Saved {args.output}")
        return

    raise ValueError("Provide either --text or --input-jsonl")


if __name__ == "__main__":
    main()