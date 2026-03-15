import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

inp_file = "project/data/confused_vs_neutral.jsonl"
out_file = "outputs/confused_test_responses.jsonl"

Path("outputs").mkdir(exist_ok=True)

with open(inp_file) as f, open(out_file,"w") as out:
    for line in f:
        item = json.loads(line)

        prompt = item["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_tokens = model.generate(
                **inputs,
                max_new_tokens=200
            )

        text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)

        out.write(json.dumps({
            "id": item["id"],
            "prompt": prompt,
            "response": text
        })+"\n")

print("Done. Responses saved to outputs/")
