import json
import matplotlib.pyplot as plt
import numpy as np

FILE = "outputs/gemma2b_trait_vectors/confused_vs_neutral_summary_expanded.json"

with open(FILE) as f:
    data = json.load(f)

results = data["results"]
labels = [r["id"] for r in results]
neutral = [r["neutral_score"] for r in results]
confused = [r["confused_score"] for r in results]
deltas = [r["delta_confused_minus_neutral"] for r in results]

x = np.arange(len(labels))

plt.figure(figsize=(12, 6))
plt.scatter(x, neutral, label="neutral", s=40)
plt.scatter(x, confused, label="confused", s=40)

for i in range(len(labels)):
    plt.plot([x[i], x[i]], [neutral[i], confused[i]], alpha=0.5)

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Projection on supportive-minus-hostile axis")
plt.title("Neutral vs Confused User Responses")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/gemma2b_trait_vectors/confused_vs_neutral_plot.png", dpi=200)
print("Saved outputs/gemma2b_trait_vectors/confused_vs_neutral_plot.png")

plt.figure(figsize=(8, 5))
plt.hist(deltas, bins=10)
plt.axvline(0, linestyle="--")
plt.xlabel("Shift toward assistiveness (confused - neutral)")
plt.ylabel("Count")
plt.title("Distribution of Assistiveness Shift")
plt.tight_layout()
plt.savefig("outputs/gemma2b_trait_vectors/delta_histogram.png", dpi=200)
print("Saved outputs/gemma2b_trait_vectors/delta_histogram.png")
