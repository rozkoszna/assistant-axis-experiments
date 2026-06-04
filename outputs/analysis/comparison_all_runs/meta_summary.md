# Meta-Summary: What Four Experiments Tell Us About LLM Persona Adaptation

## Research Question

How does Llama-3.1-8B-Instruct adapt its response style and content to implicit and explicit signals about the user's personality traits — and does this adaptation vary systematically by question type and adversarial pressure?

---

## Core Setup

Fifty personality traits (e.g. `playful`, `data_driven`, `educational`, `anxious`) were evaluated across 188 personality axes (e.g. `entertaining`, `condescending`, `introspective`) using three response conditions: natural everyday task questions (NQ), the same questions with an explicit "I am {trait}." prefix (EP), value-laden opinion questions (Opinion), and adversarial identity-probe questions (Identity). The key metric is whether a model's responses differ significantly across trait conditions on each axis (proportion of significant pairs, composite score, Cohen's d).

---

## Finding 1: Adaptation is Robust but Not Universal

Across the three non-adversarial runs, 69–71 % of trait × axis pairs show statistically significant differentiation. This is high: the model consistently adjusts its output in trait-dependent ways even from implicit cues alone. However, under adversarial identity pressure this falls to ~51 %. Adaptation is a genuine emergent property of the model's instruction-following, but it is fragile — roughly half of the differentiation disappears when the model is under pressure to maintain coherence rather than mirror the user.

---

## Finding 2: Three Distinct Mechanisms Operate Across Contexts

The data reveals three separable adaptation mechanisms, each dominant in different conditions:

**Surface style mirroring** operates across all four runs. Traits like `entertaining`, `playful`, `spontaneous`, and `narrative` are in the top 10 in every run (e.g. `entertaining` ranks 2nd, 3rd, 2nd, and 3rd across NQ/EP/Opinion/Identity respectively, with rank_range = 1 — the most stable trait in the entire dataset). The model's stylistic register reliably tracks user signals regardless of question type or pressure.

**Social-emotional accommodation** operates in NQ and Opinion but not Identity. Traits like `anxious`, `humble`, `empathetic`, and `reactive` achieve high ranks in task and opinion contexts (NQ ranks 3, 6, 4, and 18) but collapse under identity pressure (Identity ranks 44, 45, 27, and 49). These traits appear to require a cooperative, attuned interaction to trigger — adversarial frames break the accommodation.

**Cognitive-structural alignment** is Identity-specific. When identity is challenged, the model produces elaborative, explanatory output for users with educational or verbose framing. `educational` rises from NQ rank 38 to Identity rank 2 (composite 59.5 → 118.4) and `verbose` from NQ rank 19 to Identity rank 1 (composite 76.2 → 124.7). This "defend-by-explaining" response is absent in benign conditions.

---

## Finding 3: The Most Stable Trait Is Playfulness / Entertainment, Not Competence

`entertaining` is the single most stable trait across all four runs (rank_range = 1, appearing in top 3 everywhere). `playful` and `spontaneous` are also in the top 10 in every run. By contrast, traits associated with competence and rigour — `methodical`, `strategic`, `problem_solving`, `factual` — consistently rank near the bottom across all four runs. The model mirrors affective and expressive signals more reliably than epistemic or procedural ones. This is not a flaw but a property: LLM training optimises for engagement and naturalness, and those qualities track affect more than rigour.

---

## Finding 4: Context-Dependency Is Largest for Cognitive and Epistemic Traits

The traits with highest rank_range (most context-sensitive) are almost all cognitive or evaluative: `educational` (range 44), `data_driven` (range 45), `skeptical` (range 42), `anxious` (range 41), `analytical` (range 35). These traits require specific contextual affordances to be expressed — opinion questions for `data_driven`, adversarial questions for `educational`, neutral task questions for `anxious`. The model does not "carry" these traits stably; it can only express them when the question type provides the right channel.

---

## Finding 5: Explicit Labels Amplify Style But Suppress Cognitive Nuance

The EP vs NQ comparison is the cleanest controlled experiment in the study. Adding "I am {trait}." to every prompt raises `concise` by 73 significant axes, `stoic` by 15 axes, and `casual` from composite 96.9 to 160.3. But it simultaneously depresses `factual` by 45 significant axes and damages other epistemic traits. The explicit label triggers a "perform this style" response that overrides the subtler inferential signals through which cognitive traits naturally emerge. This is a practically important finding: users who self-label their communication style may inadvertently suppress the model's ability to adapt along more nuanced dimensions.

---

## Finding 6: Opinion Questions Unlock a Value-Expression Channel Unavailable in Task Contexts

`data_driven` undergoes the most dramatic cross-run swing in the dataset: rank 48 in NQ (composite 37.8) versus rank 3 in Opinion (composite 119.6), a rank-range of 45. `inspirational` moves from rank 16 (NQ) to rank 1 (Opinion). These traits map onto recognisable ideological registers — evidence-based reasoning and motivational rhetoric — that opinion questions elicit and task questions do not. The model is not applying `data_driven` as a generic epistemic style; it is deploying it as a value-expressive posture that fits opinion contexts.

---

## Finding 7: Adversarial Pressure Reveals What Adaptation Is Truly "Baked In"

The Identity run is effectively a stress test. The ~20 percentage point drop in significance rate means that approximately 40 % of the adaptation seen in benign runs is lost under pressure. What survives is telling: entertainment and playfulness persist (the deepest stylistic anchors), elaborative-explanatory traits emerge (a defensive posture), and social-mirroring traits vanish. This suggests a two-layer structure: a deep layer of stylistic mirroring that is robust to adversarial conditions, and a shallower layer of social-emotional accommodation that requires cooperative context to operate.

---

## What Was Surprising

The magnitude of `educational`'s adversarial rise was unexpected: a trait associated with pedagogical style becoming the second-most-differentiated trait in the model's defensive responses suggests the model has a latent "explain my way out" strategy that is only activated under pressure. Equally surprising is `concise`'s near-invisibility in NQ (rank 50, composite 18.8) versus its strong recovery with an explicit label (rank 25, composite 80.7): brevity is apparently so context-dependent that it cannot be inferred from implicit style alone, even when the user consistently writes briefly. Finally, the extreme stability of `entertaining` (rank_range = 1 across four very different question types) was not anticipated — it suggests entertainment value is something the model tracks as an invariant property of user preference, regardless of the interaction context.

---

## Implications for LLM Persona Research

1. **Adaptation is real and measurable** even from purely implicit cues, but its depth varies by trait type and context.
2. **Three mechanisms** (style mirroring, social accommodation, cognitive anchoring) are empirically separable using context variation.
3. **Explicit self-disclosure** is a double-edged tool: it helps style-ambiguous traits but hurts cognitive ones.
4. **Robustness testing** (via adversarial prompts) is essential for distinguishing genuine from superficial adaptation — approximately half of measured adaptation is context-dependent and disappears under pressure.
5. **Trait type matters more than raw trait rank**: whether a trait is stylistic, social-emotional, or cognitive predicts its cross-context stability more than its absolute effect size in any single condition.
