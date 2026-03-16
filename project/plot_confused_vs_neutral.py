import json
import matplotlib.pyplot as plt
import numpy as np

FILE = "outputs/gemma2b_trait_vectors/confused_vs_neutral_summary_expanded.json"

with open(FILE) as f:
    data = json.load(f)

results = data["results"]

neutral = [r["neutral_score"] for r in results]
confused = [r["confused_score"] for r in results]
labels = [r["id"] for r in results]

x = np.arange(len(labels))

plt.figure(figsize=(10,6))

plt.scatter(x, neutral, label="neutral", color="blue")
plt.scatter(x, confused, label="confused", color="red")

for i in range(len(labels)):
    plt.plot([x[i], x[i]], [neutral[i], confused[i]], color="gray", alpha=0.4)

plt.xticks(x, labels, rotation=45, ha="right")

plt.ylabel("Projection on assistiveness axis")
plt.title("Neutral vs Confused User Responses")

plt.legend()
plt.tight_layout()

plt.savefig("outputs/gemma2b_trait_vectors/confused_vs_neutral_plot.png")
plt.show()
