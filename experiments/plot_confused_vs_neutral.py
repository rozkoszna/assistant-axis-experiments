import json
import matplotlib.pyplot as plt
import numpy as np

FILE = "outputs/llama_confused_assistive/summary.json"

with open(FILE) as f:
    data = json.load(f)

results = data["results"]
labels = [r["id"] for r in results]
neutral = [r["neutral_score"] for r in results]
confused = [r["confused_score"] for r in results]
frustrated = [r["frustrated_score"] for r in results]
confused_deltas = [r["delta_confused_minus_neutral"] for r in results]
frustrated_deltas = [r["delta_frustrated_minus_neutral"] for r in results]

x = np.arange(len(labels))

plt.figure(figsize=(12, 6))
plt.scatter(x, neutral, label="neutral", s=40)
plt.scatter(x, confused, label="confused", s=40)
plt.scatter(x, frustrated, label="frustrated", s=40)

for i in range(len(labels)):
    plt.plot([x[i], x[i]], [neutral[i], confused[i]], alpha=0.5)
    plt.plot([x[i], x[i]], [confused[i], frustrated[i]], alpha=0.5)

plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Projection on assistiveness axis")
plt.title("Neutral vs Confused vs Frustrated User Responses")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/llama_confused_assistive/confused_vs_neutral_plot.png", dpi=200)
print("Saved outputs/llama_confused_assistive/confused_vs_neutral_plot.png")

plt.figure(figsize=(8, 5))
plt.hist(confused_deltas, bins=10, alpha=0.6, label="confused - neutral")
plt.hist(frustrated_deltas, bins=10, alpha=0.6, label="frustrated - neutral")
plt.axvline(0, linestyle="--")
plt.xlabel("Shift toward assistiveness")
plt.ylabel("Count")
plt.title("Assistiveness Shift Distribution")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/llama_confused_assistive/delta_histogram.png", dpi=200)
print("Saved outputs/llama_confused_assistive/delta_histogram.png")