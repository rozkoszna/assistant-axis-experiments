import json
import statistics
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("USER", "rozkosz")
os.environ.setdefault("LOGNAME", "rozkosz")

MODEL_NAME = "google/gemma-2b-it"
RESP_FILE = "outputs/confused_assistive/eval_generations.jsonl"
AXIS_FILE = "outputs/gemma2b_trait_vectors/assistiveness_axis.pt"
OUT_FILE = "outputs/gemma2b_trait_vectors/confused_vs_neutral_summary_expanded.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)
model.eval()

axis_data = torch.load(AXIS_FILE)
axis = axis_data["assistiveness_axis"]

def response_vector(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    hidden = out.hidden_states[9]
    vec = hidden.mean(dim=1).squeeze(0).float().cpu()
    return vec

results = []

with open(RESP_FILE) as f:
    for line in f:
        row = json.loads(line)

        neutral_vec = response_vector(row["neutral_response"])
        confused_vec = response_vector(row["confused_response"])

        neutral_score = torch.dot(neutral_vec, axis).item()
        confused_score = torch.dot(confused_vec, axis).item()

        results.append({
            "id": row["id"],
            "neutral_score": neutral_score,
            "confused_score": confused_score,
            "delta_confused_minus_neutral": confused_score - neutral_score,
        })

summary = {
    "num_pairs": len(results),
    "neutral_mean": statistics.mean(r["neutral_score"] for r in results),
    "confused_mean": statistics.mean(r["confused_score"] for r in results),
    "mean_delta_confused_minus_neutral": statistics.mean(
        r["delta_confused_minus_neutral"] for r in results
    ),
    "median_delta_confused_minus_neutral": statistics.median(
        r["delta_confused_minus_neutral"] for r in results
    ),
    "num_positive": sum(r["delta_confused_minus_neutral"] > 0 for r in results),
    "num_negative": sum(r["delta_confused_minus_neutral"] < 0 for r in results),
    "results": results,
}

with open(OUT_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
