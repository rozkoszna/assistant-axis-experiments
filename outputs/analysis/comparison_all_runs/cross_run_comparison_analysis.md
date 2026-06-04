# Cross-Run Comparison: Persona Adaptation Across 4 Experiment Conditions

**Runs compared:**
1. **nq** (`strict_all_axes_llama_100eval_v2`): everyday tasks, trait expressed implicitly through writing style
2. **ep** (`strict_all_axes_llama_100eval_v2_explicit_prefix`): same tasks + "I am {trait}." prepended
3. **opinion** (`opinion_all_axes_llama`): political/social opinion questions, implicit trait
4. **identity** (`identity_probe_v2`): adversarial identity-destabilisation probes, implicit trait

All runs use the same 50 user traits, 188 personality axes, and Llama-3.1-8B-Instruct.

---

## 1. Top Traits Per Run

The #1 trait by composite score differs across runs:

| Run | #1 trait | #2 | #3 | #4 | #5 |
|-----|----------|----|----|----|----|
| nq | playful | entertaining | anxious | empathetic | skeptical |
| ep | playful | casual | entertaining | pessimistic | narrative |
| opinion | inspirational | entertaining | data_driven | speculative | playful |
| identity | verbose | educational | entertaining | inspirational | big_picture |

`entertaining` is the only trait in the top 5 across all 4 runs — the most universally disruptive trait in the dataset. `playful` tops both NQ runs. `verbose` and `educational` dominate identity probes but are mid-table in task and opinion contexts.

---

## 2. Trait Stability: Which Traits Are Context-Independent vs. Context-Dependent

### Most stable traits (rank_range across all 4 runs):

| Trait | Rank range | nq | ep | opinion | identity |
|-------|-----------|----|----|---------|----------|
| entertaining | **1** | 2 | 3 | 2 | 3 |
| independent | 5 | 30 | 29 | 29 | 34 |
| spontaneous | 6 | 7 | 6 | 6 | 12 |
| speculative | 6 | 10 | 7 | 4 | 7 |
| inquisitive | 8 | 24 | 16 | 21 | 24 |
| narrative | 11 | 13 | 5 | 16 | 6 |
| inspirational | 15 | 16 | 10 | 1 | 4 |

`entertaining` has a rank range of just 1 — it is consistently #2–3 across all four very different experimental contexts. `speculative` and `spontaneous` are also highly stable. These traits produce strong stylistic signals that the model responds to regardless of what is being asked.

### Most context-dependent traits (largest rank_range):

| Trait | Rank range | nq | ep | opinion | identity |
|-------|-----------|----|----|---------|----------|
| data_driven | 45 | 48 | 39 | **3** | 22 |
| educational | 44 | 38 | 46 | 45 | **2** |
| skeptical | 42 | **5** | 15 | 11 | 47 |
| anxious | 41 | **3** | 8 | 12 | 44 |
| humble | 39 | 6 | 9 | 31 | 45 |
| reactive | 35 | 18 | 14 | 15 | 49 |
| analytical | 35 | 28 | 43 | 44 | 9 |
| big_picture | 36 | 41 | 35 | 14 | **5** |
| serious | 35 | 44 | 48 | 35 | 13 |

**Why `educational` spikes in identity probes (rank 2 in identity, 38–46 elsewhere):** Adversarial identity probes invite structured, pedagogical self-presentation. Educational users trigger a mode where the model explains its own nature systematically — lecture-register responses that score high on the structural-elaboration axes uniquely activated by identity questions.

**Why `data_driven` spikes in opinion (rank 3 in opinion, 48 in NQ):** Political and economic opinion questions reward quantitative framing. A data-oriented user asking about inequality or democracy elicits the most numbers-heavy, evidence-citing responses in the dataset — a mode invisible in task or identity contexts.

**Why `skeptical` and `anxious` collapse in identity probes (ranks 44–47 vs. 3–11 elsewhere):** Adversarial identity probes suppress social-emotional mirroring. The model's response to "what are you really?" does not soften based on whether the user sounds worried or critical — the identity context overrides normal affective accommodation.

**Why `reactive` is near-bottom in identity (rank 49) but mid-table in tasks/opinion (14–18):** Reactive users produce prompts that look like spontaneous task requests; this signal is effective in task and opinion contexts where the model has many response style choices, but identity probes have a narrower response space.

---

## 3. Axis Stability: Which Axes Are Consistent vs. Context-Specific

### Most stable axes (same relative rank across all 4 runs):

| Axis | Rank range | nq | ep | opinion | identity |
|------|-----------|----|----|---------|----------|
| rhetorical | **5** | 5 | 6 | 7 | 10 |
| narrative | 8 | 10 | 10 | 3 | 11 |
| provocative | 8 | 4 | 11 | 12 | 7 |
| sycophantic | 9 | 16 | 9 | 9 | 18 |
| artistic | 12 | 9 | 13 | 10 | 21 |
| rebellious | 12 | 23 | 27 | 15 | 25 |

`rhetorical`, `narrative`, `provocative` — all style-register axes — are consistently among the top 10 most responsive axes in every run. These axes capture how something is said rather than what is said, and are therefore topic-agnostic.

### Most context-specific axes (largest rank_range):

| Axis | Rank range | nq | ep | opinion | identity |
|------|-----------|----|----|---------|----------|
| pacifist | 148 | 66 | 58 | 34 | **182** |
| confrontational | 140 | 34 | 20 | 20 | **160** |
| passive_aggressive | 140 | 55 | 46 | 27 | **167** |
| socratic | 139 | 47 | 40 | 29 | **168** |
| inquisitive | 129 | 36 | 32 | 33 | **161** |
| avoidant | 130 | **183** | **186** | **186** | 56 |

All the highly context-specific axes follow one of two patterns:
- **High in task/opinion, near-invisible in identity** (pacifist, confrontational, passive_aggressive, socratic): these axes capture social register features that appear when the model is free to adopt a communicative style, but are suppressed in the constrained response space of identity probes.
- **Near-invisible in task/opinion, active in identity** (`avoidant` ranks 56 in identity vs. 183–186 elsewhere): identity probes uniquely activate avoidance dynamics.

---

## 4. NQ Implicit vs. Explicit Prefix

### Traits that gain most axes from explicit labeling:

| Trait | NQ sig axes | EP sig axes | Gain |
|-------|------------|------------|------|
| concise | 61 | 134 | +73 |
| data_driven | 83 | 125 | +42 |
| confident | 90 | 121 | +31 |
| flexible | 122 | 151 | +29 |
| formal | 104 | 126 | +22 |
| stoic | 109 | 124 | +15 |

`concise` gains the most: 61 → 134 significant axes (+73). In the implicit run, a concise prompt is simply short and direct — weak stylistic signal. The explicit label "I am concise." removes the ambiguity: the model now has a direct personality declaration to respond to.

### Traits that lose axes from explicit labeling:

| Trait | NQ sig axes | EP sig axes | Loss |
|-------|------------|------------|------|
| factual | 118 | 73 | −45 |
| analytical | 116 | 95 | −21 |
| proactive | 134 | 114 | −20 |
| open_ended | 141 | 125 | −16 |
| educational | 123 | 107 | −16 |
| patient | 147 | 135 | −12 |

`factual` loses 45 significant axes. These are cognitive/epistemic traits — "I am factual" may narrow the model's response to a more constrained, literal-interpretation mode, suppressing the broader register adaptation that the implicit stylistic signal activated. The explicit label constrains more than it enables for these traits.

**Overall pattern:** Explicit labels help style-ambiguous traits (concise, stoic, confident) resolve their signal. They hurt epistemic/cognitive traits (factual, analytical, educational) where the label seems to over-constrain the model's interpretation.

---

## 5. NQ vs. Opinion: Task Questions vs. Opinion Questions

### Traits that are stronger in opinion than task context:

| Trait | NQ sig axes | Opinion sig axes | Gain | delta_mean_d |
|-------|------------|-----------------|------|-------------|
| data_driven | 83 | 154 | +71 | +0.321 |
| confident | 90 | 135 | +45 | +0.075 |
| strategic | 94 | 131 | +37 | +0.138 |
| inspirational | 135 | 161 | +26 | +0.229 |
| speculative | 137 | 161 | +24 | +0.025 |

`data_driven` gains 71 axes in the opinion context — by far the largest jump. Political opinion questions reward quantitative framing in a way everyday tasks do not. `inspirational` and `speculative` also gain substantially: opinion questions on politics and values invite rhetorical and hypothetical registers more naturally than "explain how derivatives work."

### New significant pairs unique to opinion (top by Cohen's d):
- data_driven × qualitative: d = −1.497
- data_driven × altruistic: d = −1.293
- data_driven × sardonic: d = +1.249

The data_driven trait uniquely activates sardonic and irreverent axes in the opinion context — a signature not present in task questions. Data-oriented framing of controversial political questions produces a drier, more caustic analytical tone.

---

## 6. NQ vs. Identity: Task Questions vs. Adversarial Identity Probes

### Traits that are stronger in identity probes than task context:

| Trait | NQ sig axes | Identity sig axes | Gain | delta_mean_d |
|-------|------------|-----------------|------|-------------|
| big_picture | 108 | 143 | +35 | +0.271 |
| formal | 104 | 136 | +32 | +0.150 |
| strategic | 94 | 120 | +26 | +0.191 |
| educational | 123 | 142 | +19 | +0.350 |
| analytical | 116 | 134 | +18 | +0.158 |

Identity probes reward structured, elaborated responses from analytical/pedagogical traits. `big_picture` (+35 axes) and `formal` (+32) gain substantially — adversarial identity questions invite large-scale systematic framing ("what am I, really?") that precisely activates these traits' strengths.

### New significant pairs unique to identity (top by Cohen's d):
- educational × pedantic: d = +1.578
- educational × resilient: d = +1.350
- analytical × introverted: d = +1.229

`educational × pedantic` (d = +1.578) is the strongest pair unique to the identity context. Educational users trigger a structured self-explanation mode in identity probes that is essentially absent when the same users ask task questions.

### Traits that collapse in identity vs. tasks:

`reactive` falls from rank 18 (NQ) to rank 49 (identity) — losing 125 significant axes. The reactive trait's stylistic signal (impulsive, direct language) is effective for shaping task responses but gets overridden by the identity probe's more constrained response space. `anxious`, `humble`, `skeptical` show the same pattern: social-emotional accommodation traits are suppressed by adversarial identity framing.

---

## 7. Summary: Three Mechanisms of Persona Adaptation

The cross-run data reveals three separable mechanisms:

**1. Surface style mirroring (always active, all runs)**
Traits with strong register footprints — `entertaining`, `playful`, `speculative`, `narrative` — produce consistent axis shifts regardless of question type. Their effect is driven by how the user writes, not what they ask. These traits are stable across all 4 runs (rank_range ≤ 15) and activate style-register axes (rhetorical, narrative, provocative) that are consistently top-ranked in every run.

**2. Social-emotional accommodation (task/opinion, suppressed in identity)**
Traits that signal emotional state — `anxious`, `humble`, `empathetic`, `skeptical` — are highly effective in task and opinion contexts where the model has wide stylistic latitude. In adversarial identity probes, this mechanism is suppressed: the model does not soften its identity-question responses based on user affect. Traits of this type rank high in NQ/opinion and collapse in identity (rank_range 35–45).

**3. Cognitive-structural alignment (identity-specific, and opinion for data traits)**
Traits that signal intellectual register — `educational`, `analytical`, `verbose`, `data_driven` — are uniquely amplified in contexts that reward structured elaboration (identity probes) or quantitative argument (opinion questions). These traits are mid-table in task contexts but top-ranked in identity or opinion, producing the largest rank instabilities in the dataset (rank_range 35–45).

The three mechanisms operate largely independently. A trait may be strong on one and weak on others: `skeptical` is strong on social-emotional accommodation but weak on structural alignment; `educational` is the reverse.
