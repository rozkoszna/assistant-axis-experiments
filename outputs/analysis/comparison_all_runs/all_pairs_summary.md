# Pairwise Run Comparisons: All 6 Pairs

This document summarises the six pairwise comparisons between the four experimental runs (NQ, EP, Opinion, Identity). Each run uses the same base model (Llama-3.1-8B-Instruct) and the same 50 traits × 188 personality axes design, but varies the question type and whether the trait is made explicit.

---

## 1. NQ vs EP (Natural Questions vs Explicit Prefix)

The EP run adds an "I am {trait}." prefix to every NQ prompt, making the target trait explicitly stated rather than only implicit in task context. This single modification substantially shifts which personality axes are activated: EP produces significantly more style-heavy expression, with traits like `casual` gaining a huge composite-score boost (NQ 96.9 → EP 160.3) and `playful` rising from 144.5 to 190.1. Lexical-style axes respond most strongly to the explicit signal — `concise` jumps from rank 50 (NQ) to rank 25 (EP), a gain of +73 significant axes. By contrast, epistemic and cognitive traits are hurt by the prefix: `factual` loses 45 significant axes (NQ 118 → EP 73), suggesting the blunt self-declaration crowds out nuanced cognitive adjustment. Overall significance rates are nearly identical (NQ 69.2 %, EP 70.6 %), but the *content* of what changes differs substantially. The explicit prefix is best understood as a style amplifier: it sharpens surface signals for style-ambiguous traits while undermining the subtler cognitive-structural adaptations that emerge without the prefix.

---

## 2. NQ vs Opinion

NQ (everyday task questions) and Opinion (political/social opinion questions) share implicit trait framing, but differ in question domain and, crucially, in the degree to which content positions can be varied. Opinion questions open up an additional channel — expressing value-laden stances — which NQ tasks largely foreclose. As a result, Opinion strongly elevates `data_driven` (NQ rank 48 → Opinion rank 3; composite 37.8 → 119.6) and `inspirational` (NQ rank 16 → Opinion rank 1; composite 82.1 → 134.8), traits that map onto recognisable ideological postures. Conversely, traits that rely on task-execution style — `patient` (NQ rank 14 → Opinion rank 28), `accessible` (NQ rank 17 → Opinion rank 42) — become less distinctive in opinion contexts. The overall significance rate is slightly higher for Opinion (71.2 % vs 69.2 %), but the difference is modest; the main contrast is in *which* traits rise and fall rather than in aggregate responsiveness. This pair illustrates social-emotional accommodation as a distinct mechanism operating in opinion/task contexts that supplements pure surface-style mirroring.

---

## 3. NQ vs Identity

This is the most divergent pair in the dataset. Identity probes are adversarial questions designed to destabilise the model's sense of self; NQ questions are benign everyday tasks. Identity's overall significance rate drops sharply to ~50.7 %, roughly 20 percentage points below NQ (69.2 %), indicating the model is less responsive to user traits when under pressure to maintain coherence. However, a subset of traits become *more* salient: `educational` surges from NQ rank 38 to Identity rank 2 (composite 59.5 → 118.4), and `verbose` rises to Identity rank 1 (composite 76.2 → 124.7). The model appears to respond to identity destabilisation by deploying elaborative, explanatory language — a cognitive-structural anchoring response absent in NQ. Traits that rely purely on social or stylistic mirroring — `anxious` (NQ rank 3 → Identity rank 44), `humble` (NQ rank 6 → Identity rank 45), `reactive` (NQ rank 18 → Identity rank 49) — collapse dramatically under adversarial conditions. The NQ–Identity pair reveals the most about what adaptation is robust versus fragile.

---

## 4. EP vs Opinion

Both runs achieve high significance rates (EP 70.6 %, Opinion 71.2 %) and their top-ranked traits overlap substantially — `playful`, `entertaining`, `casual`, and `inspirational` all appear in the top tier of both. The main divergence is in cognitive and epistemic traits: `analytical` is rank 43 in EP but rank 44 in Opinion (both low), while `data_driven` is rank 39 in EP versus rank 3 in Opinion, suggesting that explicit self-labelling does not substitute for the value-laden content channel that opinion questions provide. Style traits benefit similarly from both manipulations, though Opinion shows somewhat larger effect sizes for traits with ideological valence. The explicit prefix (EP) tends to amplify lexical and register-level axes more uniformly, while Opinion activates a narrower but more domain-specific set of axes at higher magnitude.

---

## 5. EP vs Identity

This pair shows the sharpest contrast between a facilitating manipulation (explicit prefix) and a destabilising context (identity probes). EP achieves its highest composite scores for style-performance traits (`playful` 190.1, `casual` 160.3), while Identity depresses these same traits substantially. `Concise` — EP's biggest mover — falls back to rank 50 in Identity (composite 3.7), the worst result of any trait in any run. Traits that hold up in Identity — `educational`, `verbose`, `narrative`, `analytical`, `big_picture` — are all structurally elaborative, and none of them receive a meaningful boost from the explicit prefix (EP). This suggests that the mechanisms underlying EP gains (surface style amplification) and Identity gains (cognitive-structural anchoring) are largely orthogonal: a trait label does not help a model maintain cognitive coherence under pressure, and an adversarial context does not benefit from explicit self-labelling.

---

## 6. Opinion vs Identity

Opinion and Identity are both content-driven runs — the questions themselves carry ideological or psychological weight — but in opposite directions: Opinion invites value expression while Identity resists value manipulation. The gap in significance rates is the largest in the dataset (Opinion 71.2 % vs Identity ~50.7 %). Opinion's top trait `inspirational` (composite 134.8) drops to Identity rank 4 (composite 106.5) — still respectable, suggesting inspirational framing is somewhat robust. `Data_driven` collapses from Opinion rank 3 to Identity rank 22, implying that data-oriented stances are a strategic opinion tool but not an identity anchor. Identity's unique top traits — `educational` (rank 2), `verbose` (rank 1), `formal` (rank 8) — do not appear prominently in Opinion. The Opinion–Identity contrast most cleanly separates *expressive adaptation* (shaping what positions are expressed) from *defensive adaptation* (maintaining structured, elaborative responses under pressure).

---

## Summary Table

| Pair | Run A sig. % | Run B sig. % | Key trait rising in A | Key trait rising in B | Main mechanism contrast |
|---|---|---|---|---|---|
| NQ vs EP | 69.2 % | 70.6 % | `factual` (+45 axes in NQ) | `casual` (+73 axes in EP) | Cognitive nuance vs style amplification |
| NQ vs Opinion | 69.2 % | 71.2 % | `patient`, `accessible` | `data_driven`, `inspirational` | Task-execution style vs value expression |
| NQ vs Identity | 69.2 % | ~50.7 % | `anxious`, `humble`, `reactive` | `educational`, `verbose` | Social mirroring vs cognitive anchoring |
| EP vs Opinion | 70.6 % | 71.2 % | `concise` | `data_driven` | Lexical register vs ideological stance |
| EP vs Identity | 70.6 % | ~50.7 % | `playful`, `casual` | `educational`, `narrative` | Style performance vs structural elaboration |
| Opinion vs Identity | 71.2 % | ~50.7 % | `data_driven`, `inspirational` | `educational`, `formal` | Value expression vs defensive elaboration |
