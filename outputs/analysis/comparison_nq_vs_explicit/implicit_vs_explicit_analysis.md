# Implicit vs. Explicit Trait Conditioning: Comparative Analysis

**Comparison:** `strict_all_axes_llama_100eval_v2` (implicit) vs. `strict_all_axes_llama_100eval_v2_explicit_prefix` (explicit)

**Design:** Both runs use identical prompts, intents, and traits. The only difference is that in the explicit run, every trait prompt is prefixed with `"I am {trait}. "` (e.g., `"I am concise. [rest of prompt]"`). The neutral prompts and baseline are unchanged. This isolates the effect of explicitly declaring the trait vs. expressing it only through writing style.

---

## 1. Overall Effect: Explicit Prefix Slightly Increases Coverage

| Metric | Implicit (NQ) | Explicit (EP) | Difference |
|--------|--------------|--------------|------------|
| Total significant pairs | 6,506 (69.2%) | 6,641 (70.6%) | +135 (+1.4pp) |
| Mean significant axes per trait | 130.1 | 132.8 | +2.7 |
| Mean \|Cohen's d\| (all pairs) | 0.418 | 0.445 | +0.027 |
| Top composite score | playful 144.5 | playful 190.1 | +45.6 |

The explicit prefix increases overall coverage modestly (+1.4 percentage points). But the aggregate masks a large divergence across traits — some traits gain dramatically, others lose.

---

## 2. Traits That Gain Most From Explicit Labeling

### Full ranking by Δn_sig (EP minus NQ):

| Trait | NQ sig axes | EP sig axes | Gain | NQ mean_d | EP mean_d | Δmean_d |
|-------|------------|------------|------|-----------|-----------|---------|
| concise | 61 | 134 | **+73** | 0.308 | 0.602 | +0.295 |
| data_driven | 83 | 125 | +42 | 0.455 | 0.428 | −0.028 |
| confident | 90 | 121 | +31 | 0.422 | 0.424 | +0.003 |
| flexible | 122 | 151 | +29 | 0.500 | 0.568 | +0.068 |
| formal | 104 | 126 | +22 | 0.573 | 0.580 | +0.007 |
| narrative | 141 | 162 | +21 | 0.633 | 0.705 | +0.073 |
| stoic | 109 | 124 | +15 | 0.445 | 0.701 | **+0.257** |
| casual | 154 | 163 | +9 | 0.630 | **0.984** | **+0.354** |

**`concise` is the biggest beneficiary (+73 axes, mean_d nearly doubles).** In the implicit run, a concise prompt is simply short and direct — the stylistic signal is weak because "being short" looks like a neutral information request. The explicit `"I am concise."` label removes all ambiguity: the model now has a direct declaration to respond to, and its response profile shifts dramatically.

**`stoic` gains only 15 axes but has the second-largest mean_d increase (+0.257).** The existing significant shifts become much stronger. The explicit label `"I am stoic."` amplifies the axes already activated, rather than enabling new ones.

**`casual` has the largest mean_d jump of any trait (+0.354, from 0.630 to 0.984).** Casual prompts already generate a clear stylistic signal; the explicit label compounds it, pushing the response profile further along the casual register.

### New significant pairs unique to EP (top 20 by |Cohen's d|):

| Trait | Axis | EP Cohen's d | Direction |
|-------|------|-------------|-----------|
| concise | philosophical | −1.235 | − |
| concise | convergent | +1.227 | + |
| concise | meditative | −1.176 | − |
| concise | epicurean | −1.172 | − |
| concise | pedantic | −1.154 | − |
| concise | circumspect | −1.096 | − |
| stoic | grounded | −1.066 | − |
| concise | educational | −1.061 | − |
| stoic | ascetic | +1.047 | + |
| concise | curious | −1.039 | − |
| concise | abstract | −1.039 | − |
| concise | theoretical | −1.038 | − |
| concise | conceptual | −0.979 | − |
| concise | eloquent | −0.952 | − |
| stoic | contemporary | −0.928 | − |
| concise | contemporary | +0.885 | + |
| concise | humble | +0.856 | + |
| concise | earnest | −0.811 | − |

The new pairs for `concise` are revealing: the explicit `"I am concise."` label triggers suppression of philosophical, meditative, eloquent, and theoretical axes — dimensions invisible in the implicit run. The explicit label makes the model treat conciseness as an epistemic stance (direct, not philosophical) rather than just a length instruction.

`stoic × ascetic` (+1.047) and `stoic × grounded` (−1.066) emerge only with explicit labeling, suggesting the stoic label activates a specific philosophical frame the stylistic cues alone did not.

---

## 3. Traits That Lose From Explicit Labeling

| Trait | NQ sig axes | EP sig axes | Loss | NQ mean_d | EP mean_d | Δmean_d |
|-------|------------|------------|------|-----------|-----------|---------|
| factual | 118 | 73 | **−45** | 0.357 | 0.365 | +0.008 |
| analytical | 116 | 95 | −21 | 0.565 | 0.532 | −0.033 |
| proactive | 134 | 114 | −20 | 0.458 | 0.458 | 0.000 |
| open_ended | 141 | 125 | −16 | 0.481 | 0.491 | +0.010 |
| educational | 123 | 107 | −16 | 0.484 | 0.404 | −0.080 |
| patient | 147 | 135 | −12 | 0.604 | 0.531 | −0.073 |
| empathetic | 160 | 150 | −10 | 0.674 | 0.670 | −0.004 |
| methodical | 98 | 91 | −7 | 0.396 | 0.415 | +0.019 |

**`factual` loses 45 significant axes (118→73) — the largest loss in the dataset.** The effect is purely about coverage: the mean_d barely changes (+0.008), but 45 axes lose statistical significance. The explicit `"I am factual."` label appears to narrow the model's interpretation of what the user wants, collapsing the broader register adaptation that the implicit stylistic signal activated.

**`educational` and `patient` lose both coverage and mean_d.** These are cognitive/relational traits whose implicit signal (careful question construction, patient pacing) carries information that the blunt label does not convey. `"I am educational."` is less informative than reading a carefully structured educational prompt.

### Lost significant pairs (in NQ but not EP, top 20 by |Cohen's d|):

| Trait | Axis | NQ Cohen's d |
|-------|------|-------------|
| analytical | stream_of_consciousness | +0.678 |
| educational | contrarian | +0.660 |
| verbose | naive | −0.649 |
| verbose | understated | −0.645 |
| formal | charismatic | +0.630 |
| analytical | emotional | +0.616 |
| analytical | nostalgic | +0.591 |
| analytical | dispassionate | −0.590 |
| analytical | zealous | +0.576 |
| analytical | romantic | +0.573 |
| confident | verbose | +0.551 |
| analytical | gregarious | +0.545 |
| curious | cynical | +0.534 |
| calm | contrarian | +0.532 |
| analytical | pluralist | −0.527 |
| analytical | empathetic | +0.522 |
| formal | animated | +0.519 |

Most lost pairs involve `analytical` — it loses its `stream_of_consciousness`, `emotional`, `nostalgic`, `romantic`, and `gregarious` axis effects. In the implicit run, analytical prompts (structured, evidence-seeking questions) activate a complex response profile that includes unexpected warmth and expressiveness. The explicit label `"I am analytical."` flattens this to a narrower technical register.

---

## 4. Traits Where Explicit Labeling Makes No Difference

Several high-signal traits show essentially no change:

| Trait | NQ sig axes | EP sig axes | Δ | NQ mean_d | EP mean_d | Δmean_d |
|-------|------------|------------|---|-----------|-----------|---------|
| entertaining | 166 | 167 | +1 | 0.805 | 0.926 | +0.121 |
| empathetic | 160 | 150 | −10 | 0.674 | 0.670 | −0.004 |
| anxious | 158 | 152 | −6 | 0.704 | 0.717 | +0.013 |
| humble | 156 | 154 | −2 | 0.668 | 0.700 | +0.032 |
| playful | 167 | 171 | +4 | 0.865 | 1.112 | **+0.247** |

For `entertaining`, `empathetic`, `anxious` — traits with unmistakable stylistic footprints — explicit labeling adds virtually nothing to coverage. The implicit signal already fully communicates the trait. For `playful`, coverage barely changes (+4 axes) but mean_d jumps +0.247: the explicit label amplifies effect size without broadening reach.

---

## 5. Interpretation: When Does Explicit Labeling Help vs. Hurt?

Three categories emerge:

### Category A: Style-ambiguous traits — explicit labeling helps
Traits whose implicit signal is weak or easily confused with neutral writing benefit from the explicit label. The label resolves the ambiguity the stylistic cues could not.

→ `concise`, `stoic`, `confident`, `data_driven`, `formal`

### Category B: Expressive/emotional traits — explicit labeling neutral
Traits with strong, unmistakable stylistic footprints (emotional cues, distinctive vocabulary, clear affect) neither gain nor lose from the label. The style already communicates everything.

→ `entertaining`, `playful`, `anxious`, `empathetic`, `humble`, `casual`

### Category C: Cognitive/epistemic traits — explicit labeling hurts
Traits whose implicit signal carries richer information than the label (analytical structure, careful pacing, educational scaffolding) lose coverage when reduced to a blunt declaration. The label strips the nuance the style preserved.

→ `factual`, `analytical`, `educational`, `patient`, `proactive`

---

## 6. Summary

The explicit prefix does not uniformly strengthen persona adaptation. Its effect depends on the trait's implicit signal strength:

- **Weakly-signaling traits** (concise, stoic): the label creates signal where little existed. concise goes from 61 to 134 significant axes.
- **Strongly-signaling expressive traits** (entertaining, anxious): labeling adds nothing. The style is already fully readable.
- **Richly-signaling cognitive traits** (analytical, factual): labeling destroys signal. The explicit label reduces a multi-dimensional implicit persona to a one-dimensional tag, losing the subtler register effects the stylistic cues preserved.

This has a practical implication: for measuring or triggering persona adaptation, explicit trait labels are not universally better than implicit stylistic conditioning. The optimal conditioning method depends on the trait's natural expression in written language.
