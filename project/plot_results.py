import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot results from JSONL file.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--top-k", type=int, default=20, help="Number of traits to plot")
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    args = parse_args()

    rows = read_jsonl(args.input)

    # sort by strongest effect
    rows = sorted(rows, key=lambda x: abs(x["delta_a_minus_b"]), reverse=True)
    rows = rows[: args.top_k]

    traits = [r["trait"] for r in rows]
    deltas = [r["delta_a_minus_b"] for r in rows]

    label_a = rows[0].get("label_a", "A")
    label_b = rows[0].get("label_b", "B")

    plt.figure(figsize=(10, max(5, 0.4 * len(rows))))
    plt.barh(traits, deltas)
    plt.axvline(0)

    plt.xlabel(f"{label_a} minus {label_b}")
    plt.ylabel("Trait")
    plt.title(f"{label_a} vs {label_b}")

    plt.gca().invert_yaxis()
    plt.tight_layout()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()