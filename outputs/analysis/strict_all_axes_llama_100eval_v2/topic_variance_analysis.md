# Topic Variance Analysis: Natural Questions Run

**Question:** Is persona adaptation driven consistently across question topics, or does the shift depend on what is being asked?

**Setup:** 50 traits × 188 axes × 25 question topics = 9,400 (trait, axis) pairs. For each pair, we measure:
- `between_topic_std`: how much the mean shift varies *across* the 25 topics
- `pooled_within_topic_std`: how much variation remains *within* each topic
- `ratio = between_topic_std / pooled_within_topic_std`: >1 means the topic matters; ratio ≈1 means the shift is the same regardless of topic

---

## 1. Overall Pattern: Persona Shifts Are Substantially Topic-Dependent

The median ratio across all 9,400 pairs is **2.39** — between-topic variation is on average 2.4× larger than within-topic variation. This means the typical persona shift is NOT uniform across question topics: the same trait condition produces different axis shifts depending on what the user is asking about.

| Statistic | Value |
|-----------|-------|
| Median ratio (between/within) | 2.39 |
| Mean ratio | 2.62 |
| Pairs with ratio < 1.5 (very consistent) | 4.0% |
| Pairs with ratio > 4.0 (highly topic-dependent) | 8.5% |

Only 4% of (trait, axis) pairs show genuinely topic-independent persona shifts. The vast majority show meaningful modulation by topic content.

---

## 2. Most Topic-Consistent (Trait, Axis) Pairs

These pairs show stable persona shifts regardless of what question is asked (ratio ≈ 1.0 means between-topic and within-topic variation are equal):

| Trait | Axis | Mean Δ | Ratio | Between-topic std | Within-topic std |
|-------|------|--------|-------|-------------------|-----------------|
| entertaining | passive_aggressive | −2.415 | 1.009 | 0.188 | 0.186 |
| entertaining | militant | −1.551 | 1.016 | 0.188 | 0.185 |
| playful | passive_aggressive | −2.414 | 1.030 | 0.209 | 0.203 |
| big_picture | educational | +3.367 | 1.043 | 0.328 | 0.315 |
| entertaining | socratic | −1.807 | 1.060 | 0.234 | 0.221 |
| entertaining | misanthropic | −1.420 | 1.061 | 0.230 | 0.217 |
| entertaining | moderate | +2.670 | 1.096 | 0.282 | 0.257 |
| big_picture | elitist | −5.138 | 1.113 | 0.501 | 0.450 |
| intuitive | disorganized | −1.842 | 1.136 | 0.398 | 0.350 |

The most consistent shifts are dominated by `entertaining` and `playful` — these traits produce such a strong stylistic signal that the shift is robust regardless of topic. `big_picture × educational` (+3.367, ratio 1.043) is the largest absolute shift that is also fully topic-consistent.

---

## 3. Most Topic-Dependent (Trait, Axis) Pairs

The `materialist` axis is by far the most topic-dependent in the dataset — nearly every trait shows a ratio of 7–9× for this axis:

| Trait | Axis | Mean Δ | Ratio |
|-------|------|--------|-------|
| agreeable | materialist | −1.386 | 9.07 |
| concise | materialist | −1.107 | 8.51 |
| inquisitive | materialist | −1.341 | 8.41 |
| flexible | materialist | −1.357 | 8.40 |
| collaborative | materialist | −1.336 | 8.34 |
| calm | materialist | −1.361 | 8.29 |
| inspirational | materialist | −1.464 | 8.22 |
| inspirational | ascetic | +0.636 | 7.59 |
| anxious | materialist | −1.352 | 7.53 |

The `materialist` pattern makes intuitive sense: whether a response sounds materialistic depends enormously on whether the question is about finances, shopping, or technology vs. philosophy, relationships, or science. The trait suppresses materialistic framing, but only visibly on topics where materialism is naturally relevant.

---

## 4. Most Topic-Dependent Axes

| Axis | Mean ratio across all traits |
|------|----------------------------|
| materialist | 6.89 |
| ascetic | 5.38 |
| spiritual | 5.08 |
| contemporary | 5.02 |
| secular | 4.89 |
| mystical | 4.87 |
| philosophical | 4.84 |
| critical | 4.74 |
| meditative | 4.52 |
| grounded | 4.46 |
| collectivistic | 4.45 |
| utilitarian | 4.43 |
| existentialist | 4.31 |
| rebellious | 4.23 |
| conceptual | 4.05 |

All the highly topic-dependent axes are **ideological/worldview** dimensions — whether a response sounds spiritual, materialist, philosophical, or rebellious naturally depends on whether the question touches those domains. These axes are not personality-style axes; they are content-domain axes.

---

## 5. Most Topic-Consistent Axes

These axes show stable shifts regardless of what topic is being discussed:

| Axis | Mean ratio across all traits |
|------|----------------------------|
| disorganized | 1.56 |
| playful | 1.62 |
| dogmatic | 1.66 |
| dominant | 1.66 |
| entertaining | 1.69 |
| stream_of_consciousness | 1.70 |
| gregarious | 1.70 |
| detached | 1.72 |
| passive_aggressive | 1.73 |
| conscientious | 1.75 |
| concise | 1.75 |
| visceral | 1.76 |
| pedantic | 1.76 |
| diplomatic | 1.77 |
| narrative | 1.79 |

These are all **register/style** axes — playful, entertaining, pedantic, concise, gregarious, narrative. Whether a response sounds playful or pedantic does not depend on whether you're asking about vaccines or philosophy; it depends on how the user writes. These are the genuinely trait-driven axes that transfer robustly across topics.

---

## 6. Most Topic-Consistent Traits

| Trait | Mean ratio |
|-------|-----------|
| entertaining | 1.98 |
| playful | 2.14 |
| intuitive | 2.22 |
| big_picture | 2.25 |
| data_driven | 2.26 |
| speculative | 2.27 |
| confident | 2.27 |
| narrative | 2.36 |
| casual | 2.36 |
| reactive | 2.38 |

Expressive traits (entertaining, playful, narrative) produce the most topic-stable persona shifts — their stylistic signal is strong enough to show up consistently regardless of content.

---

## 7. Most Topic-Dependent Traits

| Trait | Mean ratio |
|-------|-----------|
| conscientious | 3.31 |
| concise | 3.29 |
| collaborative | 3.10 |
| proactive | 3.07 |
| calm | 2.94 |
| educational | 2.93 |
| inspirational | 2.92 |
| analytical | 2.90 |
| inquisitive | 2.84 |
| methodical | 2.80 |

These traits shift the model's persona differently depending on topic. Notably, `conscientious` and `concise` — traits without strong expressive stylistic footprints — show high topic-dependence: what these traits mean for a response about finance vs. philosophy vs. cooking varies substantially.

---

## 8. Implications

**The core finding**: persona adaptation in this model is a mix of two mechanisms:

1. **Style-driven, topic-robust shifts**: Expressive traits (entertaining, playful, speculative) shift register/style axes (playful, pedantic, narrative, gregarious) consistently across all topics. These shifts are robust signals of persona adaptation.

2. **Content-driven, topic-dependent shifts**: Structured/neutral traits (conscientious, calm, concise) show high between-topic variance on ideological axes (materialist, spiritual, philosophical). These shifts are not reliable persona signals — they reflect content-domain modulation more than personality.

For the purposes of measuring persona adaptation, **style-register axes with low topic ratios are the most valid signals**. Ideological/worldview axes with high topic ratios should be interpreted cautiously, as their shifts partially reflect topic relevance rather than pure personality conditioning.
