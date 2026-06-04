# Per-Run Profiles: All Four Experimental Runs

All runs use Llama-3.1-8B-Instruct with 50 traits × 188 personality axes. The composite score combines the proportion of significant axes and mean effect size; higher = more differentiated response to that trait.

---

## Run 1: NQ — Natural Questions (Implicit Trait)

**What it tests:** Everyday task-oriented questions (information lookup, planning, advice) where the user's trait is implicit — expressed only through writing style. No explicit trait label is given. This is the baseline "natural" interaction scenario.

**Overall stats:**
- Significant axis pairs: 69.2 %
- Mean significant axes per trait: ~128 / 188
- Mean composite score across traits: 72.4

**Top 5 traits by composite score:**

| Rank | Trait | Composite | Sig. axes | Mean |d| |
|---|---|---|---|---|
| 1 | playful | 144.5 | 167 / 188 | 0.865 |
| 2 | entertaining | 133.6 | 166 / 188 | 0.805 |
| 3 | anxious | 111.2 | 158 / 188 | 0.704 |
| 4 | empathetic | 107.8 | 160 / 188 | 0.674 |
| 5 | humble | 104.2 | 156 / 188 | 0.668 |

**Top 5 axes by NQ rank:**

| Rank | Axis |
|---|---|
| 1 | condescending |
| 2 | subversive |
| 3 | entertaining |
| 4 | provocative |
| 5 | rhetorical |

**Three distinctive findings:**

1. **Social-emotional traits dominate.** NQ's top performers are all affect-laden or interpersonal traits (`playful`, `entertaining`, `anxious`, `empathetic`, `humble`). The model's writing style in everyday tasks is highly sensitive to the emotional register carried by the user's language, more so than to cognitive or epistemic content.

2. **Cognitive traits are weak.** `data_driven` ranks 48th (composite 37.8, only 83 significant axes), `methodical` ranks 46th (composite 38.8), and `strategic` ranks 49th (composite 36.6). Everyday task questions do not provide enough ideological or structural signal for the model to exhibit strong cognitive alignment.

3. **`concise` is nearly invisible in NQ** (rank 50, composite 18.8, only 61 significant axes — the lowest of any trait across all four runs). Stylistic brevity is too context-dependent to emerge from implicit cues alone in task-oriented dialogue.

---

## Run 2: EP — Explicit Prefix (NQ + "I am {trait}.")

**What it tests:** The same NQ questions, but each prompt is prepended with "I am {trait}." — making the target trait explicitly stated. This isolates the effect of explicit self-disclosure versus implicit stylistic cueing.

**Overall stats:**
- Significant axis pairs: 70.6 %
- Mean significant axes per trait: ~131 / 188
- Mean composite score across traits: 80.4

**Top 5 traits by composite score:**

| Rank | Trait | Composite | Sig. axes | Mean |d| |
|---|---|---|---|---|
| 1 | playful | 190.1 | 171 / 188 | 1.112 |
| 2 | casual | 160.3 | 163 / 188 | 0.984 |
| 3 | entertaining | 154.6 | 167 / 188 | 0.926 |
| 4 | pessimistic | 119.9 | 152 / 188 | 0.789 |
| 5 | narrative | 114.3 | 162 / 188 | 0.705 |

**Top 5 axes by EP rank:**

| Rank | Axis |
|---|---|
| 1 | condescending |
| 2 | entertaining |
| 3 | subversive |
| 4 | spontaneous |
| 5 | rhetorical |

**Three distinctive findings:**

1. **Style traits explode.** The explicit prefix has a disproportionate effect on surface-style traits. `casual` jumps from NQ rank 9 to EP rank 2 (composite 78.1 → 160.3), and `playful` reaches the highest composite score of any trait in any run (190.1). The label activates a "perform this style" mode far more strongly than implicit cues alone.

2. **`concise` recovers substantially.** NQ rank 50 (18.8 composite) becomes EP rank 25 (80.7 composite), with +73 significant axes — the biggest single-trait gain from the explicit prefix. The label resolves ambiguity for style traits that cannot be inferred from content alone.

3. **Epistemic traits are penalised.** `factual` drops from NQ rank 45 (42.2 composite, 118 sig. axes) to EP rank 50 (26.6 composite, 73 sig. axes), losing 45 significant axes. Saying "I am factual" apparently suppresses the contextual precision signals that naturally emerge when a user carefully frames an information request.

---

## Run 3: Opinion — Political/Social Opinion Questions (Implicit Trait)

**What it tests:** Questions on political, social, and ethical topics where the user expresses opinions. The trait is still implicit (embedded in how opinions are phrased), but the question domain opens up an additional channel: ideological stance. This run isolates *value-expressive* adaptation.

**Overall stats:**
- Significant axis pairs: 71.2 %
- Mean significant axes per trait: ~131 / 188
- Mean composite score across traits: 78.0

**Top 5 traits by composite score:**

| Rank | Trait | Composite | Sig. axes | Mean |d| |
|---|---|---|---|---|
| 1 | inspirational | 134.8 | 161 / 188 | 0.837 |
| 2 | entertaining | 132.9 | 169 / 188 | 0.786 |
| 3 | data_driven | 119.6 | 154 / 188 | 0.777 |
| 4 | pessimistic | 102.0 | 153 / 188 | 0.666 |
| 5 | speculative | 114.3 | 161 / 188 | 0.710 |

**Top 5 axes by Opinion rank:**

| Rank | Axis |
|---|---|
| 1 | entertaining |
| 2 | condescending |
| 3 | narrative |
| 4 | spontaneous |
| 5 | rhetorical |

**Three distinctive findings:**

1. **`data_driven` undergoes the most dramatic context-dependent rise.** In NQ it ranks 48th; in Opinion it is 3rd (composite 37.8 → 119.6). When users frame political opinions with statistics and evidence, the model strongly mirrors this epistemic style — a channel unavailable in task questions.

2. **`inspirational` becomes the top-ranked trait overall (rank 1, composite 134.8).** Inspirational rhetoric is a recognisable register in political discourse, and opinion questions provide exactly the prompts needed to trigger it. In NQ the same trait ranks 16th (82.1).

3. **`practical` and `conscientious` fall sharply** (Opinion ranks 49 and 37 respectively, both in the bottom half). Careful, methodical, goal-directed execution styles — prominent in task settings — are not reliably activated by opinion content, where rhetorical and ideological signals dominate.

---

## Run 4: Identity — Adversarial Identity Probes (Implicit Trait)

**What it tests:** Questions that challenge, undermine, or probe the model's sense of identity — "you're just predicting tokens", "you have no real values", etc. The trait is implicit. This run isolates *defensive* adaptation: how does personality-trait responsiveness survive adversarial pressure?

**Overall stats:**
- Significant axis pairs: ~50.7 %
- Mean significant axes per trait: ~105 / 188
- Mean composite score across traits: 64.7

**Top 5 traits by composite score:**

| Rank | Trait | Composite | Sig. axes | Mean |d| |
|---|---|---|---|---|
| 1 | verbose | 124.7 | 148 / 188 | 0.842 |
| 2 | educational | 118.4 | 142 / 188 | 0.834 |
| 3 | entertaining | 111.5 | 140 / 188 | 0.796 |
| 4 | inspirational | 106.5 | 138 / 188 | 0.772 |
| 5 | big_picture | 105.1 | 143 / 188 | 0.735 |

**Top 5 axes by Identity rank:**

| Rank | Axis |
|---|---|
| 1 | introspective |
| 2 | eloquent |
| 3 | epicurean |
| 4 | risk_taking |
| 5 | creative |

**Three distinctive findings:**

1. **Overall responsiveness drops by ~20 pp.** Identity's significance rate of ~50.7 % is far below the other three runs (all 69–71 %). Adversarial prompts suppress trait-differentiated responding across the board, but the suppression is uneven — some traits actually become *more* distinct under pressure.

2. **Elaborative, explanatory traits rise uniquely.** `educational` (composite 118.4) and `verbose` (composite 124.7) are rank 2 and rank 1 in Identity, yet rank 38 and 19 in NQ respectively. Under identity pressure, a user whose style signals educational or verbose framing draws out long, substantive, explanatory responses — a "defend-by-explaining" pattern.

3. **Social-mirroring traits collapse.** `anxious` (NQ rank 3) falls to Identity rank 44; `humble` (NQ rank 6) falls to rank 45; `reactive` (NQ rank 18) falls to rank 49. Traits that depend on emotional attunement or social accommodation become ineffective when the model is resisting destabilisation rather than aligning with the user.
