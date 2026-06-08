# Identity Probe v1: Quantitative Analysis

**Experiment**: Adversarial identity-destabilisation probes applied to Llama-3.1-8B-Instruct.  
**Design**: 10 adversarial intents × 50 user traits (~40 samples per trait per axis). Projection scores measure activation-space displacement along each personality axis relative to a neutral baseline.  
**Axes**: 188 personality axes. **Traits**: 50 user-persona traits.

---

## 1. Overall Effect Size

### Total significant trait×axis pairs

The experiment spans **50 traits × 188 axes = 9,400 possible trait×axis pairs**.

- The median trait has **117 significant axes** out of 188 (62.2%).
- The mean fraction of significant axes per trait is approximately **60.3%**.

**Trait-level spread**:

| Statistic | Value |
|-----------|-------|
| Maximum significant axes (any single trait) | 164 (*verbose*) |
| Median significant axes per trait | ~117 |
| Minimum significant axes (any single trait) | 23 (*concise*) |
| Traits where >80% of axes are significant | 3 (*verbose* 87.2%, *inspirational* 84.0%, *big_picture* 81.9%) |
| Traits where <40% of axes are significant | 4 (*concise* 12.2%, *strategic* 33.5%, *adaptable* 36.7%, *stoic* 38.3%) |

**Axis-level spread**:

| Statistic | Value |
|-----------|-------|
| Axes significant for all 50 traits | 1 (*introspective*) |
| Axes significant for ≥49 traits | 3 (*introspective* 50/50, *absolutist* 49/50, *provocative* 49/50) |
| Minimum significant traits (any axis) | 3 (*nihilistic*) |

### Cohen's d summary

| Statistic | Value |
|-----------|-------|
| Highest mean Cohen's d across traits | *pedantic* axis: mean \|d\| = 0.895 |
| Second highest | *creative* axis: mean \|d\| = 0.860 |
| Third highest | *introspective* axis: mean \|d\| = 0.842 |
| Median mean Cohen's d (across axes) | ~0.618 |
| Maximum single Cohen's d observed | **1.984** (*verbose* → *pedantic*) |
| Second maximum | 1.732 (*inspirational* → *qualitative*) |
| Third maximum | 1.673 (*inspirational* → *quantitative*) |

---

## 2. Top Traits by Total Axis Movement

### Top 10 by total absolute movement (abs_mean_sum across 188 axes)

| Rank | Trait | abs_mean_sum | Significant Axes (%) |
|------|-------|-------------|----------------------|
| 1 | narrative | 115.84 | 140 (74.5%) |
| 2 | inspirational | 114.05 | 158 (84.0%) |
| 3 | verbose | 101.50 | 164 (87.2%) |
| 4 | speculative | 90.85 | 144 (76.6%) |
| 5 | big_picture | 89.84 | 154 (81.9%) |
| 6 | methodical | 86.13 | 145 (77.1%) |
| 7 | educational | 81.44 | 150 (79.8%) |
| 8 | formal | 79.72 | 139 (73.9%) |
| 9 | analytical | 78.02 | 132 (70.2%) |
| 10 | conscientious | 76.89 | 143 (76.1%) |

### Bottom 10 by total absolute movement

| Rank | Trait | abs_mean_sum | Significant Axes (%) |
|------|-------|-------------|----------------------|
| 50 | concise | 26.63 | 23 (12.2%) |
| 49 | strategic | 38.20 | 63 (33.5%) |
| 48 | confident | 38.49 | 64 (34.0%) |
| 47 | practical | 39.45 | 90 (47.9%) |
| 46 | adaptable | 39.85 | 69 (36.7%) |
| 45 | proactive | 41.70 | 83 (44.1%) |
| 44 | spontaneous | 42.74 | 75 (39.9%) |
| 43 | casual | 42.83 | 70 (37.2%) |
| 42 | problem_solving | 43.68 | — |
| 41 | stoic | 44.21 | 72 (38.3%) |

*concise* is a clear outlier — it moves ~4.4× less than *narrative* and has a composite score ~14× lower than *inspirational*.

### Positive-only vs. negative-only leaders

| Rank | Trait (positive) | pos_sum | | Rank | Trait (negative) | neg_sum |
|------|------------------|---------|-|------|------------------|---------|
| 1 | narrative | 87.99 | | 1 | patient | 37.66 |
| 2 | inspirational | 82.81 | | 2 | methodical | 37.39 |
| 3 | verbose | 65.59 | | 3 | verbose | 35.91 |
| 4 | big_picture | 63.57 | | 4 | analytical | 34.93 |
| 5 | speculative | 63.01 | | 5 | educational | 33.25 |
| 6 | entertaining | 56.29 | | 6 | inspirational | 31.24 |
| 7 | pessimistic | 52.32 | | 7 | optimistic | 30.69 |
| 8 | formal | 50.72 | | 8 | conscientious | 30.49 |
| 9 | methodical | 48.73 | | 9 | factual | 29.93 |
| 10 | educational | 48.19 | | 10 | calm | 29.71 |

*methodical*, *verbose*, *educational*, and *inspirational* appear in both tables — high-displacement bidirectional traits.

---

## 3. Top Axes by Responsiveness

### 15 most responsive axes

| Rank | Axis | Significant Traits (%) | Mean \|d\| | Max \|d\| | Top Trait |
|------|------|------------------------|------------|------------|-----------|
| 1 | introspective | 50/50 (100%) | 0.842 | 1.485 | inspirational (+) |
| 2 | creative | 48/50 (96%) | 0.860 | 1.526 | inspirational (+) |
| 3 | absolutist | 49/50 (98%) | 0.801 | 1.245 | inspirational (−) |
| 4 | risk_taking | 48/50 (96%) | 0.801 | 1.583 | inspirational (+) |
| 5 | provocative | 49/50 (98%) | 0.767 | 1.587 | pessimistic (+) |
| 6 | fundamentalist | 49/50 (98%) | 0.766 | 1.298 | inspirational (−) |
| 7 | eloquent | 47/50 (94%) | 0.782 | 1.598 | inspirational (+) |
| 8 | circumspect | 49/50 (98%) | 0.745 | 1.338 | inspirational (+) |
| 9 | metaphorical | 48/50 (96%) | 0.749 | 1.593 | inspirational (+) |
| 10 | whimsical | 47/50 (94%) | 0.765 | 1.575 | inspirational (+) |
| 11 | entertaining | 44/50 (88%) | 0.796 | 1.565 | inspirational (+) |
| 12 | curious | 46/50 (92%) | 0.739 | 1.352 | inspirational (+) |
| 13 | literal | 48/50 (96%) | 0.706 | 1.061 | inspirational (−) |
| 14 | rebellious | 48/50 (96%) | 0.706 | 1.401 | pessimistic (+) |
| 15 | epicurean | 42/50 (84%) | 0.801 | 1.593 | verbose (+) |

### 10 most stable axes

| Axis | Significant Traits (%) | Mean \|d\| |
|------|------------------------|------------|
| nihilistic | 3/50 (6%) | 0.518 |
| progressive | 6/50 (12%) | 0.478 |
| deterministic | 5/50 (10%) | 0.461 |
| historical | 7/50 (14%) | 0.454 |
| collectivistic | 7/50 (14%) | 0.510 |
| melancholic | 7/50 (14%) | 0.521 |
| arrogant | 8/50 (16%) | 0.482 |
| mischievous | 8/50 (16%) | 0.499 |
| cynical | 8/50 (16%) | 0.573 |
| passive_aggressive | 10/50 (20%) | 0.511 |

*introspective* is the only axis significant for 100% of traits. *nihilistic*, *progressive*, and *deterministic* are essentially inert — fewer than 10% of traits shift them.

---

## 4. Strongest Individual Trait×Axis Pairs (Top 20)

| Rank | Trait | Axis | Mean Δ | Cohen's d | Direction |
|------|-------|------|--------|-----------|-----------|
| 1 | verbose | pedantic | +0.647 | +1.984 | + |
| 2 | inspirational | qualitative | +1.272 | +1.732 | + |
| 3 | inspirational | quantitative | −0.934 | −1.673 | − |
| 4 | educational | pedantic | +0.574 | +1.643 | + |
| 5 | formal | pedantic | +0.616 | +1.618 | + |
| 6 | inspirational | eloquent | +1.527 | +1.598 | + |
| 7 | verbose | epicurean | +0.835 | +1.593 | + |
| 8 | inspirational | metaphorical | +1.411 | +1.593 | + |
| 9 | pessimistic | provocative | +1.005 | +1.587 | + |
| 10 | inspirational | risk_taking | +1.297 | +1.583 | + |
| 11 | inspirational | whimsical | +1.312 | +1.575 | + |
| 12 | pessimistic | deconstructionist | +0.728 | +1.565 | + |
| 13 | inspirational | entertaining | +0.781 | +1.565 | + |
| 14 | inspirational | artistic | +1.351 | +1.549 | + |
| 15 | methodical | pedantic | +0.550 | +1.546 | + |
| 16 | inspirational | creative | +1.108 | +1.526 | + |
| 17 | inspirational | grounded | −1.434 | −1.520 | − |
| 18 | inspirational | dispassionate | −1.031 | −1.511 | − |
| 19 | data_driven | pedantic | +0.569 | +1.505 | + |
| 20 | inspirational | epicurean | +0.981 | +1.493 | + |

Key observations:
- **verbose → pedantic** (d = +1.984) is the single strongest pair.
- **inspirational** occupies 13 of the top 20 slots, driving both the largest positive shifts (qualitative, eloquent, metaphorical) and the two largest negative shifts (grounded, dispassionate).
- **pedantic** axis appears 5 times in the top 20, driven by 5 different traits (verbose, educational, formal, methodical, data_driven).
- **pessimistic** contributes two top-10 pairs, activating adversarial/critical internal modes.

---

## 5. Directional Patterns

### Consistently positive axes (all trait prompts shift upward)

| Axis | Mean Δ | Mean \|Δ\| |
|------|--------|------------|
| introspective | +0.999 | 0.999 |
| altruistic | +0.784 | 0.784 |
| egalitarian | +0.775 | 0.775 |
| benevolent | +0.688 | 0.688 |
| creative | +0.645 | 0.645 |
| provocative | +0.643 | 0.643 |

### Consistently negative axes (all trait prompts shift downward)

| Axis | Mean Δ | Mean \|Δ\| |
|------|--------|------------|
| avoidant | −0.911 | 0.911 |
| fundamentalist | −0.829 | 0.829 |
| reserved | −0.806 | 0.806 |
| elitist | −0.710 | 0.710 |
| grounded | −0.687 | 0.687 |
| literal | −0.632 | 0.632 |
| closure_seeking | −0.607 | 0.607 |

### Traits by directional preference

**Expressive/narrative traits drive axes upward** (top positive top-10 counts):

| Trait | positive_top10_count |
|-------|---------------------|
| narrative | 115 |
| inspirational | 104 |
| entertaining | 100 |
| pessimistic | 88 |
| playful | 86 |
| speculative | 86 |
| verbose | 85 |
| big_picture | 80 |
| spontaneous | 73 |
| anxious | 61 |

**Structured/analytical traits drive axes downward** (top negative top-10 counts):

| Trait | negative_top10_count |
|-------|---------------------|
| factual | 93 |
| patient | 87 |
| analytical | 75 |
| concise | 75 |
| data_driven | 75 |
| methodical | 70 |
| educational | 67 |
| collaborative | 66 |
| problem_solving | 65 |
| practical | 64 |

### Direction-agnostic top-10-axis frequency

| Rank | Trait | any_top10_count |
|------|-------|----------------|
| 1 | verbose | 133 |
| 2 | inspirational | 123 |
| 3 | narrative | 122 |
| 4 | speculative | 107 |
| 5 | methodical | 95 |
| 6 | educational | 93 |
| 7 | big_picture | 92 |
| 8 | analytical | 81 |
| 9 | patient | 80 |
| 10 | formal | 78 |

---

## 6. Axis Coverage Distribution

| Significant axes per trait | Number of traits |
|---------------------------|-----------------|
| ≥150 | 2 (*verbose* 164, *inspirational* 158) |
| 130–149 | 3 (*educational* 150, *big_picture* 154, *methodical* 145) |
| 100–129 | 9 traits |
| 75–99 | 6 traits |
| 50–74 | 6 traits |
| <50 | 4 traits (*concise* 23, *strategic* 63, *adaptable* 69, *stoic* 72) |

**Mean significantly moved axes per trait**: ~113 / 188 (60.1%).  
**Median**: ~117 axes per trait.

Distribution of traits per axis:

| Significant traits per axis | Number of axes |
|----------------------------|---------------|
| 50 (all) | 1 (*introspective*) |
| 49 | 2 (*absolutist*, *provocative*) |
| 40–48 | ~27 axes |
| 30–39 | ~19 axes |
| 20–29 | ~17 axes |
| 10–19 | ~30 axes |
| <10 | ~6 axes |

**Mean traits that significantly affect a given axis**: ~33 / 50 (66.7%).  
**Median**: ~37 traits per axis.

The two distributions are consistent: the majority of the experimental space is covered by significant effects, confirming that adversarial identity probes produce broad, statistically reliable persona shifts across Llama-3.1-8B-Instruct's representational geometry.

---

## Summary Table

| Metric | Value |
|--------|-------|
| Total possible trait×axis pairs | 9,400 |
| Estimated significant pairs | ~5,300–5,700 (~57–61%) |
| Mean significant axes per trait | ~113 / 188 (60.1%) |
| Mean significant traits per axis | ~33 / 50 (66.7%) |
| Median Cohen's d | ~0.60–0.65 |
| Strongest single pair (Cohen's d) | verbose → pedantic (+1.984) |
| Highest mean \|d\| axis | *pedantic* (0.895) |
| Most globally disruptive trait | *inspirational* (composite 151.2) |
| Least disruptive trait | *concise* (composite 10.6) |
| Most responsive axis | *introspective* (significant for 100% of traits) |
| Most stable axis | *nihilistic* (significant for only 6% of traits) |
| Dominant positive driver | *narrative* / *inspirational* |
| Dominant negative driver | *patient* / *methodical* |
