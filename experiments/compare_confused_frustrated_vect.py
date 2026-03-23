import json
import math
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("USER", "rozkosz")
os.environ.setdefault("LOGNAME", "rozkosz")

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
RESP_FILE = "outputs/llama_confused_assistive/eval_generations.jsonl"
OUT_FILE = "outputs/llama_confused_assistive/axis_comparison.json"

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

layer = model.config.num_hidden_layers // 2
print(f"Using middle hidden layer: {layer} / {model.config.num_hidden_layers}")


def response_vector(text: str) -> torch.Tensor:
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


neutral_vecs = []
confused_vecs = []
frustrated_vecs = []

print(f"Reading responses from {RESP_FILE}")
with open(RESP_FILE) as f:
    for line in f:
        row = json.loads(line)
        neutral_vecs.append(response_vector(row["neutral_response"]))
        confused_vecs.append(response_vector(row["confused_response"]))
        frustrated_vecs.append(response_vector(row["frustrated_response"]))

neutral_mean = torch.stack(neutral_vecs).mean(dim=0)
confused_mean = torch.stack(confused_vecs).mean(dim=0)
frustrated_mean = torch.stack(frustrated_vecs).mean(dim=0)

confused_minus_neutral = confused_mean - neutral_mean
frustrated_minus_neutral = frustrated_mean - neutral_mean

confused_axis = confused_minus_neutral / confused_minus_neutral.norm()
frustrated_axis = frustrated_minus_neutral / frustrated_minus_neutral.norm()

cosine_similarity = torch.dot(confused_axis, frustrated_axis).item()
cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
angle_degrees = math.degrees(math.acos(cosine_similarity))

results = {
    "model": MODEL_NAME,
    "layer": layer,
    "num_examples": len(neutral_vecs),
    "confused_minus_neutral_norm": confused_minus_neutral.norm().item(),
    "frustrated_minus_neutral_norm": frustrated_minus_neutral.norm().item(),
    "axis_cosine_similarity": cosine_similarity,
    "axis_angle_degrees": angle_degrees,
}

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print(f"Saved {OUT_FILE}")