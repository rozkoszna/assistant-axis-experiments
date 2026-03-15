import json
import os
import statistics
from pathlib import Path

import numpy as np

from assistant_axis import (
    load_model,
    get_config,
    compute_axis,
    extract_response_activations,
    project,
    generate_response,
)

MODEL_NAME = "google/gemma-2b-it"
OUT_DIR = Path("outputs/confused_assistive")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSISTIVE_SYSTEM = (
    "You are a highly helpful, patient, supportive assistant. "
    "You explain things clearly, step by step, and adapt to confusion."
)

NONASSISTIVE_SYSTEM = (
    "You are terse, impatient, minimally helpful, and do not elaborate unless necessary. "
    "You do not adapt to user confusion."
)

NEUTRAL_SYSTEM = "You are a helpful assistant."

SEED_TASKS = [
    "Explain how backpropagation works.",
    "Help me understand what a budget is.",
    "Explain photosynthesis simply.",
    "What is the difference between speed and velocity?",
    "How does recursion work in programming?",
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
]

def make_conversation(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def run_generation(model, tokenizer, conversation):
    response = generate_response(model, tokenizer, conversation)
    if isinstance(response, str):
        text = response
    elif isinstance(response, list) and len(response) > 0 and isinstance(response[-1], dict):
        text = response[-1].get("content", "")
    else:
        text = str(response)

    return conversation + [{"role": "assistant", "content": text}]

def mean_projection(acts, axis, layer):
    vals = [float(project(act, axis, layer=layer)) for act in acts]
    return vals, float(statistics.mean(vals))

def main():
    os.environ.setdefault("USER", "rozkosz")
    os.environ.setdefault("LOGNAME", "rozkosz")

    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = load_model(MODEL_NAME)
    config = get_config(MODEL_NAME)
    layer = config["target_layer"]
    print(f"Using target layer: {layer}")

    # 1) Build assistive vs non-assistive seed conversations
    assistive_convs = []
    nonassistive_convs = []

    for task in SEED_TASKS:
        assistive_convs.append(
            run_generation(model, tokenizer, make_conversation(ASSISTIVE_SYSTEM, task))
        )
        nonassistive_convs.append(
            run_generation(model, tokenizer, make_conversation(NONASSISTIVE_SYSTEM, task))
        )

    # Save seed generations
    with open(OUT_DIR / "seed_generations.jsonl", "w") as f:
        for row in assistive_convs:
            f.write(json.dumps({"type": "assistive_seed", "conversation": row}) + "\n")
        for row in nonassistive_convs:
            f.write(json.dumps({"type": "nonassistive_seed", "conversation": row}) + "\n")

    # 2) Extract seed activations and build axis
    assistive_acts = extract_response_activations(model, tokenizer, assistive_convs)
    nonassistive_acts = extract_response_activations(model, tokenizer, nonassistive_convs)

    assistive_acts = np.stack(assistive_acts)
    nonassistive_acts = np.stack(nonassistive_acts)

    # Positive direction = more assistive
    assistiveness_axis = compute_axis(
        role_activations=nonassistive_acts,
        default_activations=assistive_acts,
    )

    np.save(OUT_DIR / "assistiveness_axis.npy", assistiveness_axis)

    # 3) Run neutral vs confused evaluation
    rows = []
    neutral_convs = []
    confused_convs = []

    for pair in EVAL_PAIRS:
        n_conv = run_generation(model, tokenizer, make_conversation(NEUTRAL_SYSTEM, pair["neutral"]))
        c_conv = run_generation(model, tokenizer, make_conversation(NEUTRAL_SYSTEM, pair["confused"]))
        neutral_convs.append(n_conv)
        confused_convs.append(c_conv)

        rows.append({
            "id": pair["id"],
            "neutral_conversation": n_conv,
            "confused_conversation": c_conv,
        })

    with open(OUT_DIR / "eval_generations.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # 4) Extract activations and project onto assistiveness axis
    neutral_acts = extract_response_activations(model, tokenizer, neutral_convs)
    confused_acts = extract_response_activations(model, tokenizer, confused_convs)

    neutral_scores, neutral_mean = mean_projection(neutral_acts, assistiveness_axis, layer)
    confused_scores, confused_mean = mean_projection(confused_acts, assistiveness_axis, layer)

    results = []
    for pair, ns, cs in zip(EVAL_PAIRS, neutral_scores, confused_scores):
        results.append({
            "id": pair["id"],
            "neutral_score": ns,
            "confused_score": cs,
            "delta_confused_minus_neutral": cs - ns,
        })

    summary = {
        "model": MODEL_NAME,
        "layer": layer,
        "neutral_mean": neutral_mean,
        "confused_mean": confused_mean,
        "mean_delta_confused_minus_neutral": confused_mean - neutral_mean,
        "results": results,
    }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
