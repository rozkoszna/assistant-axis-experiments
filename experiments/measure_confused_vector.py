import json
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/gemma-2b-it"
RESP_FILE = "outputs/confused_test_responses.jsonl"
OUT_DIR = Path("outputs/confused_vector")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Avoid getpass/getuser issues in this container
import os
os.environ.setdefault("USER", "rozkosz")
os.environ.setdefault("LOGNAME", "rozkosz")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True
)
model.eval()

def response_vector(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs)

    # Last hidden layer, mean pooled over tokens
    hidden = out.hidden_states[-1]      # [1, seq, dim]
    vec = hidden.mean(dim=1).squeeze(0).float().cpu()
    return vec

neutral = {}
confused = {}

print("Reading responses...")
with open(RESP_FILE) as f:
    for line in f:
        row = json.loads(line)
        rid = row["id"]
        text = row["response"]

        vec = response_vector(text)

        if rid.endswith("_neutral"):
            base = rid[:-8]
            neutral[base] = {"id": rid, "vec": vec}
        elif rid.endswith("_confused"):
            base = rid[:-9]
            confused[base] = {"id": rid, "vec": vec}

common = sorted(set(neutral) & set(confused))
if not common:
    raise ValueError("No matching neutral/confused pairs found.")

print(f"Found {len(common)} matched pairs")

neutral_vecs = torch.stack([neutral[k]["vec"] for k in common])
confused_vecs = torch.stack([confused[k]["vec"] for k in common])

neutral_mean = neutral_vecs.mean(dim=0)
confused_mean = confused_vecs.mean(dim=0)

# Main vector of interest
confused_minus_neutral = confused_mean - neutral_mean
confused_minus_neutral = confused_minus_neutral / confused_minus_neutral.norm()

torch.save({
    "neutral_mean": neutral_mean,
    "confused_mean": confused_mean,
    "confused_minus_neutral": confused_minus_neutral,
    "pair_ids": common,
}, OUT_DIR / "confused_vector.pt")

print("Saved vector to:", OUT_DIR / "confused_vector.pt")

# Also compute pairwise projections onto the confused-minus-neutral vector
results = []
for k in common:
    n_proj = torch.dot(neutral[k]["vec"], confused_minus_neutral).item()
    c_proj = torch.dot(confused[k]["vec"], confused_minus_neutral).item()
    results.append({
        "id": k,
        "neutral_projection": n_proj,
        "confused_projection": c_proj,
        "delta_confused_minus_neutral": c_proj - n_proj,
    })

summary = {
    "num_pairs": len(common),
    "mean_neutral_projection": mean(r["neutral_projection"] for r in results),
    "mean_confused_projection": mean(r["confused_projection"] for r in results),
    "mean_delta_confused_minus_neutral": mean(r["delta_confused_minus_neutral"] for r in results),
    "results": results,
}

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
