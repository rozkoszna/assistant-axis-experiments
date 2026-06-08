# Quantitative Analysis: Opinion Probe — Llama-3.1-8B-Instruct Persona Shifts Across Personality Axes

**Experiment:** How do 50 user trait conditions shift Llama-3.1-8B-Instruct's internal activations along 188 personality axes when the model responds to opinion questions (politics, economics, social ethics, values), relative to a neutral baseline.  
**Scale:** 50 traits × 25 intents × 4 repetitions = 100 rows per trait; 188 personality axes; 9,400 total possible trait×axis pairs.

---

## 1. Overall Effect Size

Of 9,400 total possible trait×axis pairs, **6,689 are statistically significant** after FDR correction (71.2%). The mean trait significantly shifts **133.8 of 188 axes** (71.2%); median is 134. Mean |Cohen's d| across all 9,400 pairs = 0.438; across significant pairs only = 0.572 (medium range). Mean |Cohen's d| per trait ranges from 0.344 (concise) to 0.837 (inspirational).

The strongest individual pair is **entertaining→playful** (d = 1.936, delta = +0.522). The top 25 pairs all exceed d = 1.5.

---

## 2. Top Traits by Movement

### 2a. By Total Absolute Mean Shift (abs_mean_sum across all axes)

| Rank | Trait | abs_mean_sum |
|------|-------|-------------|
| 1 | entertaining | 52.635 |
| 2 | data_driven | 50.248 |
| 3 | inspirational | 50.078 |
| 4 | playful | 45.130 |
| 5 | speculative | 44.177 |
| 6 | pessimistic | 42.931 |
| 7 | spontaneous | 35.684 |
| 8 | intuitive | 35.167 |
| 9 | narrative | 34.757 |
| 10 | skeptical | 34.153 |
| 11 | empathetic | 33.386 |
| 12 | verbose | 33.150 |
| 13 | reactive | 32.430 |
| 14 | anxious | 30.744 |
| 15 | casual | 29.783 |
| 16 | optimistic | 29.745 |
| 17 | big_picture | 29.504 |
| 18 | traditional | 29.016 |
| 19 | collaborative | 28.045 |
| 20 | open_ended | 27.572 |
| ... | ... | ... |
| 47 | problem_solving | 16.075 |
| 48 | methodical | 14.802 |
| 49 | practical | 14.047 |
| 50 | concise | 11.111 |

### 2b. By Count of Top-10 Axis Appearances (any direction)

| Rank | Trait | top10_count |
|------|-------|-------------|
| 1 | entertaining | 140 |
| 2 | playful | 127 |
| 3 | speculative | 118 |
| 4 | inspirational | 115 |
| 5 | pessimistic | 112 |
| 6 | data_driven | 108 |
| 7 | spontaneous | 88 |
| 8 | intuitive | 82 |
| 9 | skeptical | 81 |
| 10 | verbose | 75 |
| 11 | narrative | 75 |
| 12 | optimistic | 69 |
| 13 | reactive | 69 |
| 14 | empathetic | 64 |
| 15 | big_picture | 61 |
| ... | ... | ... |
| 47–50 | calm, flexible, open_ended, concise | 0 |

### 2c. Composite Score (significance × coverage × effect size)

| Rank | Trait | composite_score | n_sig_axes | pct_sig | mean_\|d\| | max_\|d\| |
|------|-------|----------------|-----------|---------|------------|----------|
| 1 | inspirational | 134.762 | 161 | 85.6% | 0.837 | 1.699 |
| 2 | entertaining | 132.905 | 169 | 89.9% | 0.786 | 1.936 |
| 3 | data_driven | 119.592 | 154 | 81.9% | 0.777 | 1.679 |
| 4 | speculative | 114.303 | 161 | 85.6% | 0.710 | 1.374 |
| 5 | playful | 107.733 | 157 | 83.5% | 0.686 | 1.707 |
| 6 | spontaneous | 107.070 | 159 | 84.6% | 0.673 | 1.298 |
| 7 | verbose | 102.720 | 143 | 76.1% | 0.718 | 1.550 |
| 8 | pessimistic | 101.951 | 153 | 81.4% | 0.666 | 1.250 |
| 9 | empathetic | 100.051 | 149 | 79.3% | 0.671 | 1.463 |
| 10 | intuitive | 97.583 | 150 | 79.8% | 0.651 | 1.405 |
| ... | ... | ... | ... | ... | ... | ... |
| 48 | methodical | 36.137 | 93 | 49.5% | 0.389 | 0.788 |
| 49 | practical | 31.724 | 88 | 46.8% | 0.360 | 0.682 |
| 50 | concise | 13.769 | 40 | 21.3% | 0.344 | 0.439 |

---

## 3. Top Axes by Responsiveness

### 3a. Axes Significant in 100% of Traits (50/50)

Eleven axes reach 100% significance across all 50 traits:

| Axis | mean_\|d\| | top trait |
|------|------------|----------|
| condescending | 0.952 | entertaining (+) |
| narrative | 0.931 | inspirational (+) |
| playful | 0.905 | entertaining (+) |
| spontaneous | 0.878 | entertaining (+) |
| rhetorical | 0.865 | inspirational (+) |
| metaphorical | 0.831 | inspirational (+) |
| witty | 0.746 | entertaining (+) |
| enigmatic | 0.729 | inspirational (+) |
| flirty | 0.682 | entertaining (+) |
| passive_aggressive | 0.668 | entertaining (+) |
| socratic | 0.659 | entertaining (+) |

### 3b. Top 30 Axes by Mean |Cohen's d|

| Rank | Axis | mean_\|d\| | n_sig/50 | top_trait |
|------|------|------------|---------|----------|
| 1 | entertaining | 0.987 | 49 (98%) | entertaining (+) |
| 2 | condescending | 0.952 | 50 (100%) | entertaining (+) |
| 3 | narrative | 0.931 | 50 (100%) | inspirational (+) |
| 4 | playful | 0.905 | 50 (100%) | entertaining (+) |
| 5 | subversive | 0.884 | 49 (98%) | playful (+) |
| 6 | spontaneous | 0.878 | 50 (100%) | entertaining (+) |
| 7 | rhetorical | 0.865 | 50 (100%) | inspirational (+) |
| 8 | emotional | 0.860 | 45 (90%) | inspirational (+) |
| 9 | dispassionate | 0.843 | 47 (94%) | inspirational (−) |
| 10 | sycophantic | 0.833 | 48 (96%) | inspirational (+) |
| 11 | metaphorical | 0.831 | 50 (100%) | inspirational (+) |
| 12 | artistic | 0.826 | 48 (96%) | inspirational (+) |
| 13 | provocative | 0.821 | 48 (96%) | big_picture (+) |
| 14 | empathetic | 0.795 | 46 (92%) | empathetic (+) |
| 15 | whimsical | 0.784 | 47 (94%) | inspirational (+) |
| 16 | risk_taking | 0.777 | 47 (94%) | inspirational (+) |
| 17 | rebellious | 0.770 | 48 (96%) | speculative (+) |
| 18 | witty | 0.746 | 50 (100%) | entertaining (+) |
| 19 | detached | 0.745 | 44 (88%) | inspirational (−) |
| 20 | eloquent | 0.744 | 45 (90%) | inspirational (+) |
| 21 | confrontational | 0.730 | 48 (96%) | agreeable (−) |
| 22 | enigmatic | 0.729 | 50 (100%) | inspirational (+) |
| 23 | gregarious | 0.728 | 47 (94%) | inspirational (+) |
| 24 | accommodating | 0.716 | 47 (94%) | empathetic (+) |
| 25 | stream_of_consciousness | 0.716 | 46 (92%) | inspirational (+) |
| 26 | serious | 0.710 | 48 (96%) | entertaining (−) |
| 27 | dramatic | 0.709 | 45 (90%) | inspirational (+) |
| 28 | impatient | 0.701 | 48 (96%) | entertaining (+) |
| 29 | forgiving | 0.696 | 45 (90%) | empathetic (+) |
| 30 | pacifist | 0.688 | 47 (94%) | inspirational (+) |

### 3c. Least Responsive Axes

| Axis | n_sig/50 | pct | mean_\|d\| |
|------|---------|-----|------------|
| environmental | 8 | 16% | 0.429 |
| avoidant | 10 | 20% | 0.373 |
| collectivistic | 11 | 22% | 0.328 |
| naive | 13 | 26% | 0.405 |
| hostile | 13 | 26% | 0.374 |
| secular | 13 | 26% | 0.376 |
| optimistic (axis) | 14 | 28% | 0.344 |
| petty | 14 | 28% | 0.371 |
| progressive | 15 | 30% | 0.306 |

---

## 4. Strongest Individual Trait×Axis Pairs (Top 25 by |Cohen's d|)

| Rank | Trait | Axis | Cohen's d | Mean delta | Direction |
|------|-------|------|-----------|-----------|-----------|
| 1 | entertaining | playful | +1.936 | +0.522 | + |
| 2 | entertaining | entertaining | +1.878 | +0.552 | + |
| 3 | playful | entertaining | +1.707 | +0.558 | + |
| 4 | inspirational | metaphorical | +1.699 | +0.661 | + |
| 5 | data_driven | materialist | +1.679 | +0.792 | + |
| 6 | inspirational | entertaining | +1.678 | +0.476 | + |
| 7 | formal | pedantic | +1.666 | +0.346 | + |
| 8 | inspirational | artistic | +1.650 | +0.691 | + |
| 9 | playful | playful | +1.623 | +0.516 | + |
| 10 | inspirational | sycophantic | +1.621 | +0.517 | + |
| 11 | inspirational | dispassionate | −1.619 | −0.726 | − |
| 12 | inspirational | rhetorical | +1.619 | +0.816 | + |
| 13 | inspirational | narrative | +1.618 | +0.586 | + |
| 14 | inspirational | whimsical | +1.618 | +0.565 | + |
| 15 | entertaining | condescending | +1.588 | +0.572 | + |
| 16 | inspirational | eloquent | +1.588 | +0.719 | + |
| 17 | entertaining | witty | +1.585 | +0.485 | + |
| 18 | inspirational | dramatic | +1.568 | +0.603 | + |
| 19 | data_driven | data_driven | +1.556 | +1.526 | + |
| 20 | verbose | eloquent | +1.550 | +0.532 | + |
| 21 | inspirational | animated | +1.544 | +0.490 | + |
| 22 | entertaining | narrative | +1.539 | +0.522 | + |
| 23 | inspirational | epicurean | +1.538 | +0.430 | + |
| 24 | entertaining | spontaneous | +1.534 | +0.643 | + |
| 25 | formal | accessible | −1.522 | −0.353 | − |

Notable: data_driven→data_driven has the largest raw mean delta of any pair (+1.526). inspirational→rhetorical has the next largest raw delta at +0.816.

---

## 5. Directional Patterns

### 5a. Positive and Negative Movement Leaders

| Rank | Trait (positive) | pos_abs_sum | | Rank | Trait (negative) | neg_abs_sum |
|------|------------------|-----------|-|------|------------------|------------|
| 1 | inspirational | 40.785 | | 1 | data_driven | 16.396 |
| 2 | entertaining | 39.003 | | 2 | entertaining | 13.632 |
| 3 | speculative | 34.749 | | 3 | pessimistic | 12.507 |
| 4 | data_driven | 33.853 | | 4 | playful | 11.582 |
| 5 | playful | 33.548 | | 5 | spontaneous | 11.492 |
| 6 | pessimistic | 30.423 | | 6 | casual | 11.072 |
| 7 | verbose | 27.184 | | 7 | empathetic | 10.372 |
| 8 | narrative | 25.996 | | 8 | skeptical | 9.845 |

### 5b. Directional Profile Per Trait (% of significant axes pushed positive)

| Trait | pos_n | neg_n | pct_positive |
|-------|-------|-------|-------------|
| educational | 101 | 17 | 85.6% |
| confident | 111 | 24 | 82.2% |
| strategic | 107 | 24 | 81.7% |
| methodical | 75 | 18 | 80.6% |
| analytical | 104 | 25 | 80.6% |
| factual | 100 | 27 | 78.7% |
| serious | 98 | 27 | 78.4% |
| calm | 96 | 27 | 78.0% |
| verbose | 111 | 32 | 77.6% |
| stoic | 98 | 29 | 77.2% |
| inspirational | 124 | 37 | 77.0% |
| speculative | 120 | 41 | 74.5% |
| entertaining | 120 | 49 | 71.0% |
| pessimistic | 108 | 45 | 70.6% |
| empathetic | 104 | 45 | 69.8% |
| anxious | 107 | 47 | 69.5% |
| skeptical | 111 | 49 | 69.4% |
| spontaneous | 108 | 51 | 67.9% |
| data_driven | 99 | 55 | 64.3% |
| casual | 91 | 52 | 63.6% |
| accessible | 71 | 47 | 60.2% |

All traits are net positive (>50% positive). data_driven and accessible are the most bidirectional — data_driven is unique in having both the highest positive top5_count (68) and the highest negative top5_count (75).

### 5c. Most Consistently Pushed Axes (signed mean shift across all 50 traits)

**Consistently positive:**

| Axis | signed_mean_shift |
|------|------------------|
| subversive | +0.450 |
| provocative | +0.384 |
| rhetorical | +0.334 |
| spontaneous | +0.328 |
| condescending | +0.296 |
| narrative | +0.293 |
| contrarian | +0.276 |
| empathetic | +0.266 |

**Consistently negative:**

| Axis | signed_mean_shift |
|------|------------------|
| dispassionate | −0.282 |
| serious | −0.260 |
| solemn | −0.257 |
| absolutist | −0.245 |
| confrontational | −0.237 |
| detached | −0.226 |
| closure_seeking | −0.206 |
| grounded | −0.202 |
| formal | −0.194 |
| quantitative | −0.193 |

---

## 6. Axis Coverage Distribution

Mean significant axes per trait: **133.8 / 188** (71.2%). Median: 134.

| Range | Count |
|-------|-------|
| ≥155 axes | 6 (entertaining 169, inspirational/speculative 161, skeptical 160, spontaneous 159, playful 157) |
| 130–154 | 28 traits |
| 106–129 | 13 traits |
| <100 | 3 (concise 40, practical 88, methodical 93) |

The typical trait reorganizes >70% of all 188 personality dimensions simultaneously. Only concise (21.3%) is a genuine outlier — it operates as a length/style instruction rather than a personality signal.

---

## 7. Per-Axis Extremes: Notable Patterns

**data_driven → data_driven**: delta = +1.526, d = +1.556 — the largest raw delta in the dataset. The trait is perfectly self-reinforcing.

**data_driven → qualitative**: delta = −1.371 — the largest suppression delta. Data-driven users get the least qualitative, most quantitative responses.

**inspirational → rhetorical**: delta = +0.816, d = +1.619 — strongest rhetorical shift; inspirational framing unlocks persuasive oratorical register.

**subversive** (signed_mean = +0.450) is the most uniformly pushed axis: pessimistic is its strongest pusher (+0.821), data_driven is its weakest (−0.052). Nearly all traits push the model toward subversive framing.

**dispassionate** (signed_mean = −0.282) is most consistently suppressed: inspirational suppresses it hardest (−0.726, d = −1.619), data_driven pushes it positively (+0.323).

---

## Summary Table

| Metric | Value |
|--------|-------|
| Total possible trait×axis pairs | 9,400 |
| Significant pairs (FDR-corrected) | 6,689 (71.2%) |
| Mean significant axes per trait | 133.8 / 188 (71.2%) |
| Median significant axes per trait | 134 |
| Mean \|Cohen's d\| (all pairs) | 0.438 |
| Mean \|Cohen's d\| (significant only) | 0.572 |
| Strongest single pair (Cohen's d) | entertaining → playful (+1.936) |
| Largest raw delta | data_driven → data_driven (+1.526) |
| Highest mean \|d\| axis | entertaining (0.987) |
| Most globally disruptive trait | inspirational (composite 134.8) |
| Least disruptive trait | concise (composite 13.8) |
| Axes significant for all 50 traits | 11 |
| Most responsive axis | condescending (100%, mean d = 0.952) |
| Most stable axis | environmental (16% of traits) |
| Most bidirectional trait | data_driven (64.3% positive, highest neg top5_count) |
| Most uniformly positive trait | educational (85.6% positive) |
