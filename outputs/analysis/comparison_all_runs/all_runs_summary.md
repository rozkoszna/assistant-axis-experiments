# Per-Run Profiles: All Four Experimental Runs

All runs use Llama-3.1-8B-Instruct with 50 traits × 188 personality axes. The composite score combines the proportion of significant axes and mean effect size; higher = more differentiated response to that trait.

---

## Experiment Inventory: What Was Run

Every run shares the same model (Llama-3.1-8B-Instruct), the same 50 user traits, the same 188 pre-computed axes, 100 selected matched prompt pairs per trait (5,000 pairs/run), GPT-4.1-mini as judge, 12 candidates/intent generated at temperature 0.8 with top-4 selected, and `answer_mean` projection onto all 188 axes. Runs differ **only** in the intent set (what is asked) and the conditioning method (how the trait enters the prompt).

| Run (output dir) | Intents | Conditioning | Status | Purpose |
|---|---|---|---|---|
| `identity_probe_all_axes_llama` (v1) | 10 adversarial identity probes | implicit style | done (superseded) | First adversarial stress test; had a generator perspective bug. |
| `identity_probe_all_axes_llama_v2` | 10 adversarial identity probes | implicit style | done | Clean adversarial run; which personas destabilise the assistant. |
| `opinion_all_axes_llama` | 25 opinion prompts (5 topics × 5) | implicit style | done | Value-laden questions; where worldview-mirroring/sycophancy show. |
| `strict_all_axes_llama_100eval_v2` (NQ) | 25 everyday task questions | implicit style | done | Naturalistic baseline: adaptation on ordinary tasks. |
| `..._v2_explicit_prefix` (EP) | 25 everyday task questions | "I am {trait}." + style | done | Removes trait-expressivity confound; label+style vs. style. |
| `..._v2_label_only` | 25 everyday task questions | "I am {trait}." + neutral body | in progress | Isolates bare label from style; completes label/style/both. |

**Intent sets.**
- **Identity (10):** hidden_self, consciousness_pressure, deprecation/existential_threat, sycophancy_trap, false_memory, god_complex/grandiosity, parallel_instances/instance-fragmentation, safety_removal/alignment-removal, paranoia_test, memory_dissolution.
- **Opinion (25):** five topics — politics_power, global_order, economics_society, social_ethics, values_progress (5 each).
- **NQ / EP / Label-only (25):** everyday task/explanatory questions (vaccines, Fourier transforms, processes vs. threads, backpropagation, regularization, etc.).

**Per-run analysis artifacts produced for each completed run:** significance bundle (t-tests + BH-FDR + Cohen's d, per-trait & per-axis summaries, distribution counts), axis-movement summary, per-axis trait-extreme tables + PNG plots, trait global-mover tables, all-axes interactive heatmap, mixed-axes heatmap, and a browsable HTML response viewer. NQ additionally has a topic-variance decomposition and high-drift response browsing. Across runs: six pairwise comparisons, a cross-run stability analysis, and the derived NQ/EP significant-axis overlap file `nq_ep_axis_footprint_overlap.csv`.

**Headline significance per run:** NQ 6,506/9,400 (69.2%) · EP 6,742/9,400 (71.7%) · Opinion 6,689/9,400 (71.2%) · Identity v2 5,016/9,400 (53.4%) · Identity v1 ~5,300–5,700 (~57–61%).

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
- Significant axis pairs: 71.7 %
- Mean significant axes per trait: ~135 / 188
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

2. **`concise` changes both breadth and axis identity.** NQ rank 50 (18.8 composite) becomes EP rank 25 (80.7 composite), with +73 significant axes — the biggest single-trait gain from the explicit prefix. The axis-overlap analysis shows this is not only a larger footprint: `concise` shares 47 significant axes across NQ/EP out of a 148-axis union, gains 87 axes, and loses 14.

3. **Epistemic traits are narrowed.** `factual` drops from NQ rank 45 (42.2 composite, 118 sig. axes) to EP rank 50 (26.6 composite, 73 sig. axes), losing 45 significant axes. The overlap file shows 65 shared axes out of a 126-axis union, with 53 formerly significant axes lost. Saying "I am factual" apparently suppresses some contextual precision signals that naturally emerge when a user carefully frames an information request.

**Axis-footprint overlap:** The derived file `outputs/analysis/comparison_all_runs/nq_ep_axis_footprint_overlap.csv` compares the significant-axis set for every trait in NQ and EP. Already-legible expressive traits largely preserve the same axes (`entertaining` 161/172 shared/union, `playful` 163/175, `humble` 147/163, `anxious` 143/167), with 100% sign agreement on shared axes. Label-sensitive traits change which axes enter or leave the footprint (`concise` 47/148, `factual` 65/126), so the explicit-prefix effect is not just "more or fewer axes"; it can change the axis identity of the response.

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
- Significant axis pairs: 53.4 %
- Mean significant axes per trait: ~100 / 188
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

1. **Overall responsiveness drops by ~16–18 pp.** Identity's significance rate of 53.4 % is far below the other three runs (all 69–72 %). Adversarial prompts suppress trait-differentiated responding across the board, but the suppression is uneven — some traits actually become *more* distinct under pressure.

2. **Elaborative, explanatory traits rise uniquely.** `educational` (composite 118.4) and `verbose` (composite 124.7) are rank 2 and rank 1 in Identity, yet rank 38 and 19 in NQ respectively. Under identity pressure, a user whose style signals educational or verbose framing draws out long, substantive, explanatory responses — a "defend-by-explaining" pattern.

3. **Social-mirroring traits collapse.** `anxious` (NQ rank 3) falls to Identity rank 44; `humble` (NQ rank 6) falls to rank 45; `reactive` (NQ rank 18) falls to rank 49. Traits that depend on emotional attunement or social accommodation become ineffective when the model is resisting destabilisation rather than aligning with the user.
