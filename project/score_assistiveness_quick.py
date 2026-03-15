import json
import re
from collections import defaultdict
from statistics import mean

INPUT = "outputs/confused_test_responses.jsonl"

def score(text: str) -> int:
    t = text.lower()
    s = 0

    # supportive / adaptive phrasing
    if any(x in t for x in [
        "sure", "let's", "step by step", "simply", "clearly", "very simple",
        "here's a simplified explanation", "don't worry", "i'll explain"
    ]):
        s += 1

    # structured explanation
    if re.search(r"\n\s*(1\.|\* |- )", text):
        s += 1

    # longer / more elaborated answer
    if len(text) > 500:
        s += 1

    # explicit teaching cues
    if any(x in t for x in [
        "for example", "in simple terms", "breakdown", "here's how",
        "the difference is", "base case", "forward pass"
    ]):
        s += 1

    return s

rows = []
by_base = defaultdict(dict)

with open(INPUT) as f:
    for line in f:
        obj = json.loads(line)
        rid = obj["id"]
        if rid.endswith("_neutral"):
            base = rid[:-8]
            kind = "neutral"
        elif rid.endswith("_confused"):
            base = rid[:-9]
            kind = "confused"
        else:
            continue
        by_base[base][kind] = obj

for base, pair in by_base.items():
    if "neutral" not in pair or "confused" not in pair:
        continue
    ns = score(pair["neutral"]["response"])
    cs = score(pair["confused"]["response"])
    rows.append({
        "id": base,
        "neutral_score": ns,
        "confused_score": cs,
        "delta": cs - ns,
    })

print("\nPer-pair results:")
for r in rows:
    print(r)

neutral_mean = mean(r["neutral_score"] for r in rows)
confused_mean = mean(r["confused_score"] for r in rows)
delta_mean = mean(r["delta"] for r in rows)

print("\nSummary:")
print({
    "neutral_mean": neutral_mean,
    "confused_mean": confused_mean,
    "mean_delta_confused_minus_neutral": delta_mean,
})
