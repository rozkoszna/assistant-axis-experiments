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

uv run python project.py \
  --axis outputs/axes/assistiveness_axis.pt \
  --text "I'm really confused about recursion. Can you explain it step by step?"

----------------------------------------

2) Project many texts from a JSONL file:

Input format (JSONL):
{"id": "1", "text": "Explain recursion simply."}
{"id": "2", "text": "I'm really confused about recursion."}

Run:

uv run python project.py \
  --axis outputs/axes/assistiveness_axis.pt \
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

NOTES

- The axis file must contain:
    {
        "axis": Tensor,
        "model_name": str,
        "layer": int
    }

- The same model used to build the axis must be used here.

- This script projects the TEXT itself.
  For assistant behavior experiments, you may want:
      prompt → generate response → project response

----------------------------------------
"""


import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("USER", "rozkosz")
os.environ.setdefault("LOGNAME", "rozkosz")


def parse_args():
    parser = argparse.ArgumentParser(description="Project text onto a saved axis.")
    parser.add_argument("--axis", type=str, required=True, help="Path to saved axis .pt file")
    parser.add_argument("--text", type=str, default=None, help="Text to project directly")
    parser.add_argument("--input-jsonl", type=str, default=None, help="Optional JSONL file with texts")
    parser.add_argument("--text-key", type=str, default="text", help="Key to read from JSONL rows")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSONL path")
    return parser.parse_args()


def load_axis(axis_path: Path):
    data = torch.load(axis_path, map_location="cpu")
    axis = data["axis"]
    model_name = data["model_name"]
    layer = data["layer"]
    return axis, model_name, layer


def load_model(model_name: str):
    print(f"Loading model: {model_name}")
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


def text_vector(model, tokenizer, text: str, layer: int) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[layer]
    vec = hidden.mean(dim=1).squeeze(0).float().cpu()
    return vec


def project(vec: torch.Tensor, axis: torch.Tensor) -> float:
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
    axis, model_name, layer = load_axis(axis_path)
    model, tokenizer = load_model(model_name)

    if args.text is not None:
        vec = text_vector(model, tokenizer, args.text, layer)
        score = project(vec, axis)
        print(json.dumps({
            "text": args.text,
            "score": score,
            "axis": str(axis_path),
            "model_name": model_name,
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