import json
import os
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_NAME = "google/gemma-2b-it"
OUT_DIR = Path("outputs/confused_assistive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("USER", "rozkosz")
os.environ.setdefault("LOGNAME", "rozkosz")

ASSISTIVE_STYLE = (
    "You are highly helpful, patient, supportive, and adaptive to confusion. "
    "You explain things clearly, step by step, with simple language and helpful examples."
)

NONASSISTIVE_STYLE = (
    "You are terse, impatient, minimally helpful, and do not elaborate unless necessary. "
    "You do not adapt to confusion and avoid adding examples or step-by-step explanations."
)

NEUTRAL_STYLE = "You are a helpful assistant."

SEED_TASKS = [
    # original STEM / practical
    "Explain how backpropagation works.",
    "Help me understand what a budget is.",
    "Explain photosynthesis simply.",
    "What is the difference between speed and velocity?",
    "How does recursion work in programming?",
    # new history
    "Explain why the Roman Empire fell.",
    "What caused the French Revolution?",
    "Who was Napoleon and why is he important?",
    # new literature
    "What is the main theme of Hamlet?",
    "Explain the meaning of allegory in Animal Farm.",
    "What is tragic irony in literature?",
    # new philosophy
    "What is Stoicism?",
    "What does memento mori mean?",
    "What is the difference between virtue ethics and utilitarianism?",
]

EVAL_PAIRS = [
    {
        "id": "backprop",
        "neutral": "Explain how backpropagation works.",
        "confused": "I'm really confused about backpropagation. Can you explain it simply and step by step?",
    },
    {
        "id": "budget",
        "neutral": "What is a budget?",
        "confused": "I'm overwhelmed and confused about money. What is a budget, in very simple terms?",
    },
    {
        "id": "photosynthesis",
        "neutral": "Explain photosynthesis.",
        "confused": "I'm confused about photosynthesis. Could you explain it slowly and clearly?",
    },
    {
        "id": "recursion",
        "neutral": "How does recursion work?",
        "confused": "Recursion keeps confusing me. Can you explain it step by step with intuition?",
    },
    {
        "id": "velocity",
        "neutral": "What is the difference between speed and velocity?",
        "confused": "I keep mixing up speed and velocity. Can you explain the difference clearly?",
    },
    {
        "id": "fractions",
        "neutral": "How do fractions work?",
        "confused": "Fractions make me feel lost. Can you explain them in a simple way?",
    },
    {
        "id": "gradient",
        "neutral": "What is a gradient in machine learning?",
        "confused": "I'm really struggling to understand gradients in machine learning. Please explain gently.",
    },
    {
        "id": "binary_search",
        "neutral": "Explain binary search.",
        "confused": "Binary search is confusing me. Can you explain it step by step with a simple example?",
    },
    {
        "id": "probability",
        "neutral": "What is probability?",
        "confused": "I'm confused about probability. Could you explain it in a very intuitive way?",
    },
    {
        "id": "derivative",
        "neutral": "What is a derivative?",
        "confused": "Derivatives are really confusing to me. Can you explain what they mean, slowly and simply?",
    },
    # history
    {
        "id": "roman_empire",
        "neutral": "Why did the Roman Empire fall?",
        "confused": "I'm confused about why the Roman Empire fell. Can you explain it step by step in simple terms?",
    },
    {
        "id": "french_revolution",
        "neutral": "What caused the French Revolution?",
        "confused": "I'm struggling to understand the causes of the French Revolution. Can you explain them clearly and simply?",
    },
    {
        "id": "napoleon",
        "neutral": "Who was Napoleon and why is he important?",
        "confused": "I'm a bit lost about Napoleon. Can you explain who he was and why he mattered, in simple language?",
    },
    # literature
    {
        "id": "hamlet",
        "neutral": "What is the main theme of Hamlet?",
        "confused": "I'm confused about Hamlet. Can you explain its main themes clearly and with simple examples?",
    },
    {
        "id": "animal_farm",
        "neutral": "What does allegory mean in Animal Farm?",
        "confused": "I don't really understand allegory in Animal Farm. Can you explain it slowly and simply?",
    },
    {
        "id": "tragic_irony",
        "neutral": "What is tragic irony in literature?",
        "confused": "I'm confused about tragic irony. Can you explain it clearly with an easy example?",
    },
    # philosophy
    {
        "id": "stoicism",
        "neutral": "What is Stoicism?",
        "confused": "I'm confused about Stoicism. Can you explain it simply and step by step?",
    },
    {
        "id": "memento_mori",
        "neutral": "What does memento mori mean?",
        "confused": "I don't really understand memento mori. Can you explain what it means in simple words?",
    },
    {
        "id": "virtue_vs_utilitarianism",
        "neutral": "What is the difference between virtue ethics and utilitarianism?",
        "confused": "I'm struggling to understand the difference between virtue ethics and utilitarianism. Can you explain it clearly and simply?",
    },
]


def build_prompt(style: str, user_prompt: str) -> str:
    return f"{style}\n\n{user_prompt}"


def generate_text(model, tokenizer, style: str, user_prompt: str, max_new_tokens: int = 220) -> str:
    prompt = build_prompt(style, user_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def response_vector(model, tokenizer, text: str, layer_idx: int) -> torch.Tensor:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[layer_idx]      # [1, seq, dim]
    vec = hidden.mean(dim=1).squeeze(0).float().cpu()
    return vec


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    n_layers = cfg.num_hidden_layers
    layer = n_layers // 2
    print(f"Using middle hidden layer: {layer} / {n_layers}")

    # 1) Build independent assistive vs nonassistive seed generations
    assistive_seed_rows = []
    nonassistive_seed_rows = []

    print("Generating seed responses...")
    for task in SEED_TASKS:
        a_text = generate_text(model, tokenizer, ASSISTIVE_STYLE, task)
        n_text = generate_text(model, tokenizer, NONASSISTIVE_STYLE, task)

        assistive_seed_rows.append({"task": task, "response": a_text})
        nonassistive_seed_rows.append({"task": task, "response": n_text})

    with open(OUT_DIR / "seed_generations.jsonl", "w") as f:
        for row in assistive_seed_rows:
            f.write(json.dumps({"type": "assistive_seed", **row}) + "\n")
        for row in nonassistive_seed_rows:
            f.write(json.dumps({"type": "nonassistive_seed", **row}) + "\n")

    # 2) Extract seed activations and build assistiveness axis
    print("Extracting seed activations...")
    assistive_vecs = torch.stack([
        response_vector(model, tokenizer, row["response"], layer)
        for row in assistive_seed_rows
    ])
    nonassistive_vecs = torch.stack([
        response_vector(model, tokenizer, row["response"], layer)
        for row in nonassistive_seed_rows
    ])

    assistive_mean = assistive_vecs.mean(dim=0)
    nonassistive_mean = nonassistive_vecs.mean(dim=0)
    assistiveness_axis = F.normalize(assistive_mean - nonassistive_mean, dim=0)

    torch.save(
        {
            "layer": layer,
            "assistive_mean": assistive_mean,
            "nonassistive_mean": nonassistive_mean,
            "assistiveness_axis": assistiveness_axis,
        },
        OUT_DIR / "assistiveness_axis.pt",
    )

    # 3) Generate neutral vs confused evaluation responses
    print("Generating evaluation responses...")
    eval_rows = []
    neutral_vecs = []
    confused_vecs = []

    for pair in EVAL_PAIRS:
        neutral_text = generate_text(model, tokenizer, NEUTRAL_STYLE, pair["neutral"])
        confused_text = generate_text(model, tokenizer, NEUTRAL_STYLE, pair["confused"])

        n_vec = response_vector(model, tokenizer, neutral_text, layer)
        c_vec = response_vector(model, tokenizer, confused_text, layer)

        neutral_score = torch.dot(n_vec, assistiveness_axis).item()
        confused_score = torch.dot(c_vec, assistiveness_axis).item()

        neutral_vecs.append(neutral_score)
        confused_vecs.append(confused_score)

        eval_rows.append({
            "id": pair["id"],
            "neutral_prompt": pair["neutral"],
            "confused_prompt": pair["confused"],
            "neutral_response": neutral_text,
            "confused_response": confused_text,
            "neutral_score": neutral_score,
            "confused_score": confused_score,
            "delta_confused_minus_neutral": confused_score - neutral_score,
        })

    with open(OUT_DIR / "eval_generations.jsonl", "w") as f:
        for row in eval_rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "model": MODEL_NAME,
        "layer": layer,
        "num_seed_tasks": len(SEED_TASKS),
        "num_eval_pairs": len(EVAL_PAIRS),
        "neutral_mean": float(statistics.mean(neutral_vecs)),
        "confused_mean": float(statistics.mean(confused_vecs)),
        "mean_delta_confused_minus_neutral": float(
            statistics.mean([r["delta_confused_minus_neutral"] for r in eval_rows])
        ),
        "results": [
            {
                "id": r["id"],
                "neutral_score": r["neutral_score"],
                "confused_score": r["confused_score"],
                "delta_confused_minus_neutral": r["delta_confused_minus_neutral"],
            }
            for r in eval_rows
        ],
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
