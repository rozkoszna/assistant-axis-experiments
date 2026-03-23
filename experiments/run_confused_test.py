import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL = "google/gemma-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out_tokens = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id
            )

        new_tokens = out_tokens[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        out.write(json.dumps({
            "id": item["id"],
            "prompt": prompt,
            "response": text
        })+"\n")

print("Done. Responses saved to outputs/")
