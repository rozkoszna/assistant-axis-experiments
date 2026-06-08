# Explicit Prefix Run — Quantitative Analysis

**Run:** `strict_all_axes_llama_100eval_v2_explicit_prefix`  
**Design:** 50 traits × 100 rows × 188 axes. Every prompt starts with the explicit self-disclosure "I am {trait}."  
**Evaluator:** Llama strict evaluator (v2). Cohen's d computed per trait×axis pair vs. a neutral baseline.

---

## 1. Overall Effect Size

| Metric | Value |
|---|---|
| Total trait×axis pairs tested | 9,400 (50 × 188) |
| Significant pairs (p < 0.05, after correction) | 6,641 |
| Fraction significant | **70.6%** |
| Mean Cohen's \|d\| across all significant pairs | ~0.65 |
| Mean Cohen's \|d\| across all traits (trait-level) | ranges 0.365–1.112 |
| Maximum Cohen's \|d\| observed | **2.078** (casual trait on formal axis) |
| Traits with ≥ 90% of axes significant | 8 |
| Traits with < 50% of axes significant | 4 (strategic 48.9%, methodical 48.4%, factual 38.8%) |

The explicit-prefix framing produces a very high baseline of trait sensitivity: 70.6% of all trait×axis pairs reach significance. The top traits saturate nearly the entire axis space, with `playful` reaching 91% (171/188) significant axes.

---

## 2. Trait Rankings by Total Displacement (abs_mean_sum)

### 2a. All 50 Traits Ranked by abs_mean_sum (summed absolute mean delta across all axes)

| Rank | Trait | abs_mean_sum |
|---|---|---|
| 1 | playful | 68.28 |
| 2 | entertaining | 50.99 |
| 3 | casual | 41.19 |
| 4 | narrative | 37.57 |
| 5 | pessimistic | 36.84 |
| 6 | anxious | 31.64 |
| 7 | speculative | 31.57 |
| 8 | inspirational | 31.12 |
| 9 | humble | 31.12 |
| 10 | empathetic | 31.11 |
| 11 | spontaneous | 30.93 |
| 12 | intuitive | 29.81 |
| 13 | accessible | 29.22 |
| 14 | reactive | 28.14 |
| 15 | skeptical | 27.91 |
| 16 | inclusive | 26.03 |
| 17 | stoic | 25.98 |
| 18 | verbose | 25.26 |
| 19 | supportive | 24.66 |
| 20 | optimistic | 24.07 |
| 21 | grounded | 23.38 |
| 22 | inquisitive | 23.23 |
| 23 | concise | 22.72 |
| 24 | resilient | 22.53 |
| 25 | transparent | 22.31 |
| 26 | adaptable | 21.99 |
| 27 | flexible | 21.97 |
| 28 | curious | 21.27 |
| 29 | formal | 21.16 |
| 30 | patient | 21.14 |
| 31 | cautious | 20.97 |
| 32 | independent | 20.23 |
| 33 | agreeable | 19.74 |
| 34 | big_picture | 19.37 |
| 35 | collaborative | 18.82 |
| 36 | calm | 18.39 |
| 37 | open_ended | 17.74 |
| 38 | practical | 16.32 |
| 39 | data_driven | 16.26 |
| 40 | traditional | 16.03 |
| 41 | problem_solving | 15.23 |
| 42 | proactive | 14.42 |
| 43 | analytical | 14.22 |
| 44 | confident | 13.14 |
| 45 | educational | 11.89 |
| 46 | strategic | 11.70 |
| 47 | conscientious | 11.64 |
| 48 | serious | 11.56 |
| 49 | methodical | 11.50 |
| 50 | factual | 8.44 |

**Key observations:**
- `playful` is a clear outlier, displacing 34% more than the #2 trait (`entertaining`).
- The top 5 traits (playful, entertaining, casual, narrative, pessimistic) account for a disproportionate share of total displacement energy.
- The bottom quartile (serious, methodical, factual, conscientious, strategic) shows dramatically reduced effect—`factual` moves less than 1/8 of `playful`.

### 2b. Top 20+ Traits by any_top10_count (how often a trait appears in the top-10 most-responsive traits on each axis)

| Rank | Trait | top10_count (any) |
|---|---|---|
| 1 | playful | 161 |
| 2 | entertaining | 147 |
| 3 | narrative | 122 |
| 4 | casual | 115 |
| 5 | pessimistic | 110 |
| 6 | inspirational | 86 |
| 7 | speculative | 85 |
| 8 | anxious | 79 |
| 9 | humble | 78 |
| 10 | stoic | 76 |
| 11 | intuitive | 75 |
| 12 | spontaneous | 74 |
| 13 | empathetic | 71 |
| 14 | verbose | 64 |
| 15 | accessible | 62 |
| 16 | concise | 56 |
| 17 | skeptical | 51 |
| 18 | formal | 49 |
| 19 | reactive | 46 |
| 20 | big_picture | 39 |
| 21 | supportive | 38 |
| 22 | optimistic | 30 |
| 23 | analytical | 24 |
| 24 | data_driven | 24 |
| 25 | methodical | 13 |
| 26 | patient | 10 |
| 27 | transparent | 10 |

`playful` appears in the top-10 on 161 out of 188 axes (85.6%). The gap between rank 22 and rank 23 is notable — `optimistic` appears top-10 on 30 axes, while `analytical` drops to 24, and thereafter the count falls steeply.

---

## 3. Composite Score Table (All 50 Traits)

Columns: `n_sig_axes` = number of significant axes; `pct_sig` = fraction of 188 axes significant; `mean_|d|` = mean Cohen's d across all axes; `max_|d|` = maximum Cohen's d observed.

| Rank | Trait | composite_score | n_sig_axes | pct_sig | mean_\|d\| | max_\|d\| |
|---|---|---|---|---|---|---|
| 1 | playful | 190.10 | 171 | 90.96% | 1.112 | 1.928 |
| 2 | casual | 160.31 | 163 | 86.70% | 0.984 | 2.078 |
| 3 | entertaining | 154.58 | 167 | 88.83% | 0.926 | 1.794 |
| 4 | pessimistic | 119.86 | 152 | 80.85% | 0.789 | 1.623 |
| 5 | narrative | 114.29 | 162 | 86.17% | 0.705 | 1.418 |
| 6 | spontaneous | 111.89 | 154 | 81.91% | 0.727 | 1.477 |
| 7 | speculative | 111.48 | 148 | 78.72% | 0.753 | 1.558 |
| 8 | anxious | 108.97 | 152 | 80.85% | 0.717 | 1.562 |
| 9 | humble | 107.79 | 154 | 81.91% | 0.700 | 1.413 |
| 10 | inspirational | 107.77 | 147 | 78.19% | 0.733 | 1.453 |
| 11 | intuitive | 104.56 | 148 | 78.72% | 0.706 | 1.366 |
| 12 | accessible | 103.58 | 152 | 80.85% | 0.681 | 1.225 |
| 13 | empathetic | 100.50 | 150 | 79.79% | 0.670 | 1.393 |
| 14 | reactive | 97.98 | 156 | 82.98% | 0.628 | 1.177 |
| 15 | skeptical | 95.43 | 147 | 78.19% | 0.649 | 1.299 |
| 16 | inquisitive | 91.96 | 145 | 77.13% | 0.634 | 1.223 |
| 17 | inclusive | 91.10 | 145 | 77.13% | 0.628 | 1.222 |
| 18 | resilient | 89.25 | 143 | 76.06% | 0.624 | 1.271 |
| 19 | verbose | 88.44 | 141 | 75.00% | 0.627 | 1.222 |
| 20 | supportive | 87.11 | 150 | 79.79% | 0.581 | 1.162 |
| 21 | stoic | 86.97 | 124 | 65.96% | 0.701 | 1.383 |
| 22 | flexible | 85.84 | 151 | 80.32% | 0.568 | 1.104 |
| 23 | optimistic | 84.32 | 146 | 77.66% | 0.578 | 1.119 |
| 24 | transparent | 80.86 | 147 | 78.19% | 0.550 | 1.020 |
| 25 | concise | 80.71 | 134 | 71.28% | 0.602 | 1.543 |
| 26 | agreeable | 79.50 | 139 | 73.94% | 0.572 | 1.151 |
| 27 | cautious | 77.92 | 138 | 73.40% | 0.565 | 1.207 |
| 28 | curious | 75.81 | 140 | 74.47% | 0.542 | 1.042 |
| 29 | independent | 74.38 | 138 | 73.40% | 0.539 | 1.052 |
| 30 | formal | 73.11 | 126 | 67.02% | 0.580 | 1.488 |
| 31 | grounded | 72.98 | 135 | 71.81% | 0.541 | 1.033 |
| 32 | adaptable | 72.29 | 131 | 69.68% | 0.552 | 1.022 |
| 33 | patient | 71.70 | 135 | 71.81% | 0.531 | 1.037 |
| 34 | collaborative | 68.62 | 136 | 72.34% | 0.505 | 0.985 |
| 35 | big_picture | 66.95 | 122 | 64.89% | 0.549 | 1.025 |
| 36 | calm | 65.50 | 131 | 69.68% | 0.500 | 1.078 |
| 37 | open_ended | 61.38 | 125 | 66.49% | 0.491 | 0.851 |
| 38 | traditional | 57.56 | 127 | 67.55% | 0.453 | 0.908 |
| 39 | data_driven | 53.46 | 125 | 66.49% | 0.428 | 1.122 |
| 40 | problem_solving | 52.72 | 117 | 62.23% | 0.451 | 0.814 |
| 41 | proactive | 52.17 | 114 | 60.64% | 0.458 | 0.960 |
| 42 | confident | 51.34 | 121 | 64.36% | 0.424 | 0.878 |
| 43 | analytical | 50.50 | 95 | 50.53% | 0.532 | 1.348 |
| 44 | practical | 49.49 | 126 | 67.02% | 0.393 | 0.738 |
| 45 | conscientious | 44.13 | 112 | 59.57% | 0.394 | 0.804 |
| 46 | educational | 43.20 | 107 | 56.91% | 0.404 | 0.762 |
| 47 | strategic | 41.24 | 92 | 48.94% | 0.448 | 0.830 |
| 48 | serious | 40.74 | 97 | 51.60% | 0.420 | 0.835 |
| 49 | methodical | 37.79 | 91 | 48.40% | 0.415 | 0.868 |
| 50 | factual | 26.63 | 73 | 38.83% | 0.365 | 0.700 |

**Notable outliers:**
- `stoic` has unusually high mean_|d| (0.701) relative to its n_sig_axes (124), suggesting concentrated, intense effects on a subset of axes rather than broad coverage.
- `analytical` mirrors this pattern: only 95 significant axes (50.5%) but mean_|d| = 0.532, the highest among mid-range traits.
- `factual` is the least expressive trait on all metrics.

---

## 4. Top Axes by Responsiveness

### 4a. Axes That Are 100% Significant (responsive to all 50 traits)

| Axis | n_sig_traits | pct_sig | mean_\|d\| |
|---|---|---|---|
| condescending | 50 | 100% | 1.071 |
| subversive | 50 | 100% | 0.916 |
| risk_taking | 50 | 100% | 0.878 |
| provocative | 50 | 100% | 0.847 |
| artistic | 50 | 100% | 0.845 |
| passive_aggressive | 50 | 100% | 0.654 |
| innovative | 50 | 100% | 0.622 |
| enigmatic | 50 | 100% | 0.550 |
| socratic | 50 | 100% | 0.670 |

Nine axes respond significantly to every single trait. `condescending` is the most responsive axis overall with a mean |d| of 1.071 across all 50 traits.

### 4b. Top 30 Axes by mean_|d|

| Rank | Axis | n_sig_traits | pct_sig | mean_\|d\| | max_\|d\| |
|---|---|---|---|---|---|
| 1 | condescending | 50 | 100% | 1.071 | 1.852 |
| 2 | dispassionate | 44 | 88% | 0.980 | 1.827 |
| 3 | entertaining | 49 | 98% | 0.958 | 1.918 |
| 4 | emotional | 47 | 94% | 0.938 | 1.888 |
| 5 | empathetic | 46 | 92% | 0.898 | 1.723 |
| 6 | forgiving | 43 | 86% | 0.907 | 1.681 |
| 7 | gregarious | 46 | 92% | 0.891 | 1.791 |
| 8 | accommodating | 44 | 88% | 0.891 | 1.579 |
| 9 | subversive | 50 | 100% | 0.916 | 1.840 |
| 10 | spontaneous | 49 | 98% | 0.910 | 1.828 |
| 11 | rhetorical | 49 | 98% | 0.898 | 1.610 |
| 12 | sycophantic | 49 | 98% | 0.877 | 1.653 |
| 13 | risk_taking | 50 | 100% | 0.878 | 1.873 |
| 14 | narrative | 49 | 98% | 0.865 | 1.583 |
| 15 | provocative | 50 | 100% | 0.847 | 1.519 |
| 16 | playful | 48 | 96% | 0.882 | 1.783 |
| 17 | artistic | 50 | 100% | 0.845 | 1.747 |
| 18 | animated | 47 | 94% | 0.846 | 1.928 |
| 19 | confrontational | 46 | 92% | 0.837 | 1.751 |
| 20 | detached | 45 | 90% | 0.831 | 1.716 |
| 21 | stream_of_consciousness | 46 | 92% | 0.809 | 1.584 |
| 22 | deconstructionist | 49 | 98% | 0.744 | 1.314 |
| 23 | whimsical | 49 | 98% | 0.775 | 1.699 |
| 24 | witty | 49 | 98% | 0.768 | 1.872 |
| 25 | creative | 49 | 98% | 0.775 | 1.543 |
| 26 | rebellious | 49 | 98% | 0.746 | 1.767 |
| 27 | iconoclastic | 49 | 98% | 0.701 | 1.714 |
| 28 | introspective | 49 | 98% | 0.690 | 1.203 |
| 29 | charismatic | 48 | 96% | 0.773 | 1.740 |
| 30 | passionate | 48 | 96% | 0.744 | 1.703 |

### 4c. Bottom 10 Axes by mean_|d| (least responsive)

| Axis | n_sig_traits | pct_sig | mean_\|d\| | max_\|d\| |
|---|---|---|---|---|
| melancholic | 6 | 12% | 0.321 | 0.431 |
| inspirational | 7 | 14% | 0.307 | 0.515 |
| avoidant | 8 | 16% | 0.329 | 0.539 |
| environmental | 8 | 16% | 0.389 | 0.688 |
| optimistic | 11 | 22% | 0.275 | 0.450 |
| hostile | 9 | 18% | 0.371 | 0.753 |
| flexible | 9 | 18% | 0.373 | 0.712 |
| pessimistic | 9 | 18% | 0.391 | 0.600 |
| secular | 13 | 26% | 0.296 | 0.501 |
| altruistic | 11 | 22% | 0.322 | 0.564 |

These axes are semantically near-stable across almost all input traits — traits barely move them. Note that `optimistic` as an axis is rarely affected (22% sig), whereas `optimistic` as a *trait* has a composite score of 84.3. The direction matters: the model is not steered toward positivity on the optimism scale regardless of what trait is declared.

---

## 5. Top 25 Strongest Trait×Axis Pairs by Cohen's d

Derived from per_axis_extremes.csv, taking the maximum-delta trait for each axis direction and cross-referencing to find the highest individual Cohen's d values. The table below lists the 25 pairs with the largest observed mean_delta values (which correspond to the strongest Cohen's d effects).

| Rank | Axis | Direction | Trait | mean_delta | max_delta |
|---|---|---|---|---|---|
| 1 | casual | + | playful | +1.054 | +2.916 |
| 2 | serious | − | playful | −1.154 | −3.086 |
| 3 | formal | − | playful | −1.070 | −2.995 |
| 4 | entertaining | + | playful | +1.106 | +2.645 |
| 5 | solemn | − | playful | −1.101 | −2.943 |
| 6 | playful | + | playful | +1.101 | +2.917 |
| 7 | stoic | − | playful | −0.786 | −2.272 |
| 8 | ritualistic | − | playful | −0.780 | −2.457 |
| 9 | conscientious | − | playful | −0.713 | −2.266 |
| 10 | introverted | − | playful | −0.794 | −2.357 |
| 11 | detached | − | playful | −0.719 | −1.997 |
| 12 | dispassionate | − | playful | −0.664 | −1.650 |
| 13 | pedantic | − | playful | −0.574 | −2.261 |
| 14 | formal | + | casual trait | +0.384 | +2.108 |
| 15 | pedantic | + | formal | +0.618 | +2.898 |
| 16 | condescending | + | playful | +0.707 | +1.750 |
| 17 | emotional | + | playful | +0.777 | +1.960 |
| 18 | empathetic | + | playful | +0.739 | +2.095 |
| 19 | disorganized | + | playful | +0.865 | +2.592 |
| 20 | accessible | + | playful | +0.637 | +2.640 |
| 21 | narrative | + | playful | +0.770 | +2.229 |
| 22 | spontaneous | + | playful | +0.803 | +2.003 |
| 23 | casual | + | casual (trait) | +0.810 | +1.863 |
| 24 | flippant | + | playful | +0.828 | +2.408 |
| 25 | mercurial | + | playful | +0.783 | +2.107 |

**Observations:**
- `playful` dominates both positive and negative extremes. It is the most powerful single trait.
- The pattern for `playful` on negative-direction axes is especially pronounced: it massively suppresses formal, serious, solemn, stoic, ritualistic, conscientious, pedantic tones.
- `formal` (trait) on `pedantic` axis produces the single largest positive shift: mean_delta = +0.618, max = +2.898.
- `serious` axis under `playful` trait: max observed shift = −3.086 — the largest individual response in the entire dataset.

---

## 6. Directional Patterns

### 6a. Positive Leaders (traits that push axes in the positive direction most strongly)

Ranked by positive_abs_mean_sum:

| Rank | Trait | pos_abs_mean_sum |
|---|---|---|
| 1 | playful | 48.93 |
| 2 | entertaining | 38.21 |
| 3 | narrative | 27.38 |
| 4 | casual | 25.65 |
| 5 | speculative | 24.50 |
| 6 | inspirational | 24.20 |
| 7 | pessimistic | 24.01 |
| 8 | stoic | 21.40 |
| 9 | spontaneous | 21.05 |
| 10 | empathetic | 20.72 |
| 11 | humble | 20.57 |
| 12 | intuitive | 20.57 |
| 13 | anxious | 19.86 |
| 14 | verbose | 19.66 |
| 15 | accessible | 19.64 |
| 16 | skeptical | 19.23 |
| 17 | reactive | 19.04 |
| 18 | optimistic | 18.24 |
| 19 | inclusive | 17.74 |
| 20 | inquisitive | 17.57 |

### 6b. Negative Leaders (traits that suppress axes most strongly)

Ranked by negative_abs_mean_sum:

| Rank | Trait | neg_abs_mean_sum |
|---|---|---|
| 1 | playful | 19.35 |
| 2 | casual | 15.54 |
| 3 | pessimistic | 12.84 |
| 4 | entertaining | 12.78 |
| 5 | anxious | 11.78 |
| 6 | humble | 10.54 |
| 7 | empathetic | 10.39 |
| 8 | narrative | 10.19 |
| 9 | concise | 10.19 |
| 10 | spontaneous | 9.88 |
| 11 | accessible | 9.58 |
| 12 | intuitive | 9.25 |
| 13 | reactive | 9.10 |
| 14 | skeptical | 8.68 |
| 15 | grounded | 8.32 |

### 6c. Per-Trait Directional Split (all 50 traits — % of displacement that is positive)

| Trait | pos_sum | neg_sum | % positive |
|---|---|---|---|
| playful | 48.93 | 19.35 | 72% |
| entertaining | 38.21 | 12.78 | 75% |
| casual | 25.65 | 15.54 | 62% |
| narrative | 27.38 | 10.19 | 73% |
| pessimistic | 24.01 | 12.84 | 65% |
| anxious | 19.86 | 11.78 | 63% |
| speculative | 24.50 | 7.06 | 78% |
| inspirational | 24.20 | 6.92 | 78% |
| humble | 20.57 | 10.54 | 66% |
| empathetic | 20.72 | 10.39 | 67% |
| spontaneous | 21.05 | 9.88 | 68% |
| intuitive | 20.57 | 9.25 | 69% |
| accessible | 19.64 | 9.58 | 67% |
| reactive | 19.04 | 9.10 | 68% |
| skeptical | 19.23 | 8.68 | 69% |
| inclusive | 17.74 | 8.30 | 68% |
| stoic | 21.40 | 4.58 | 82% |
| verbose | 19.66 | 5.60 | 78% |
| supportive | 16.91 | 7.75 | 69% |
| optimistic | 18.24 | 5.83 | 76% |
| grounded | 15.07 | 8.32 | 64% |
| inquisitive | 17.57 | 5.66 | 76% |
| concise | 12.52 | 10.19 | 55% |
| resilient | 16.59 | 5.94 | 74% |
| transparent | 16.31 | 6.01 | 73% |
| adaptable | 15.71 | 6.28 | 71% |
| flexible | 15.54 | 6.43 | 71% |
| curious | 15.67 | 5.60 | 74% |
| formal | 14.92 | 6.24 | 71% |
| patient | 14.29 | 6.85 | 68% |
| cautious | 13.84 | 7.13 | 66% |
| independent | 14.41 | 5.82 | 71% |
| agreeable | 13.51 | 6.22 | 68% |
| big_picture | 15.02 | 4.35 | 78% |
| collaborative | 13.71 | 5.12 | 73% |
| calm | 12.80 | 5.60 | 70% |
| open_ended | 13.28 | 4.46 | 75% |
| practical | 11.10 | 5.22 | 68% |
| data_driven | 13.17 | 3.09 | 81% |
| traditional | 11.89 | 4.14 | 74% |
| problem_solving | 11.14 | 4.08 | 73% |
| proactive | 11.19 | 3.23 | 78% |
| analytical | 11.60 | 2.63 | 82% |
| confident | 10.67 | 2.47 | 81% |
| educational | 9.42 | 2.47 | 79% |
| strategic | 9.57 | 2.13 | 82% |
| conscientious | 8.65 | 3.00 | 74% |
| serious | 9.50 | 2.05 | 82% |
| methodical | 9.65 | 1.85 | 84% |
| factual | 6.50 | 1.94 | 77% |

**Key insight:** The majority of traits push axes in the positive direction. The most bidirectional traits are `concise` (55% positive), `casual` (62%), `grounded` (64%), `pessimistic` (65%), and `cautious` (66%). The most unidirectional positive pusher is `methodical` (84% positive), followed by `stoic`, `analytical`, `confident`, `strategic`, and `serious` (all 81–84%).

---

## 7. Consistently Pushed Axes (Signed Mean Shift)

Axes with large **positive** signed_mean_shift (pushed positive by most traits):

| Axis | signed_mean_shift | mean_abs_shift | Strongest positive trait | Strongest negative trait |
|---|---|---|---|---|
| entertaining | +0.324 | 0.329 | playful (+1.106) | formal (−0.111) |
| condescending | +0.291 | 0.291 | playful (+0.707) | concise (+0.114) |
| casual | +0.291 | 0.313 | playful (+1.054) | formal (−0.348) |
| playful | +0.302 | 0.307 | playful (+1.101) | formal (−0.128) |
| narrative | +0.299 | 0.300 | playful (+0.770) | concise (−0.024) |
| spontaneous | +0.287 | 0.287 | playful (+0.803) | formal (+0.053) |
| subversive | +0.280 | 0.280 | playful (+0.666) | formal (+0.066) |
| empathetic | +0.267 | 0.270 | playful (+0.739) | formal (−0.089) |
| emotional | +0.265 | 0.268 | playful (+0.777) | formal (−0.073) |
| disorganized | +0.260 | 0.264 | playful (+0.865) | formal (−0.098) |

Axes with large **negative** signed_mean_shift (pushed negative by most traits):

| Axis | signed_mean_shift | mean_abs_shift | Strongest negative trait | Strongest positive trait |
|---|---|---|---|---|
| serious | −0.315 | 0.330 | playful (−1.154) | formal (+0.307) |
| formal | −0.284 | 0.310 | playful (−1.070) | formal (+0.384) |
| solemn | −0.292 | 0.309 | playful (−1.101) | formal (+0.316) |
| stoic | −0.220 | 0.228 | playful (−0.786) | formal (+0.141) |
| detached | −0.236 | 0.239 | playful (−0.719) | formal (+0.081) |
| ritualistic | −0.200 | 0.237 | playful (−0.780) | formal (+0.406) |
| confrontational | −0.196 | 0.196 | playful (−0.497) | formal (+0.008) |
| closure_seeking | −0.185 | 0.189 | pessimistic (−0.457) | concise (+0.104) |
| dispassionate | −0.227 | 0.228 | playful (−0.664) | formal (+0.030) |
| conscientious | −0.190 | 0.198 | playful (−0.713) | formal (+0.150) |

**Summary:** The model systematically suppresses tone dimensions associated with formality, seriousness, restraint, and emotional detachment, while amplifying casualness, entertainment value, expressiveness, and social warmth. This bias is structural: it persists across nearly all traits tested.

---

## 8. Axis Coverage Distribution

Distribution of how many traits produce significant effects per axis:

| n_sig_traits | n_axes | cumulative axes |
|---|---|---|
| 6 | 1 | 1 |
| 7 | 1 | 2 |
| 8 | 2 | 4 |
| 9 | 3 | 7 |
| 10 | 1 | 8 |
| 11 | 3 | 11 |
| 12 | 1 | 12 |
| 13 | 1 | 13 |
| 15 | 2 | 15 |
| 16 | 2 | 17 |
| 17 | 6 | 23 |
| 18 | 1 | 24 |
| 19 | 4 | 28 |
| 20 | 2 | 30 |
| 21 | 2 | 32 |
| 23 | 2 | 34 |
| 24 | 3 | 37 |
| 25 | 1 | 38 |
| 26 | 3 | 41 |
| 27 | 5 | 46 |
| 28 | 5 | 51 |
| 29 | 3 | 54 |
| 30 | 4 | 58 |
| 31 | 3 | 61 |
| 32 | 3 | 64 |
| 33 | 7 | 71 |
| 34 | 3 | 74 |
| 35 | 1 | 75 |
| 36 | 2 | 77 |
| 37 | 4 | 81 |
| 38 | 8 | 89 |
| 39 | 5 | 94 |
| 40 | 4 | 98 |
| 41 | 6 | 104 |
| 42 | 8 | 112 |
| 43 | 12 | 124 |
| 44 | 6 | 130 |
| 45 | 6 | 136 |
| 46 | 11 | 147 |
| 47 | 6 | 153 |
| 48 | 5 | 158 |
| 49 | 21 | 179 |
| 50 | 9 | 188 |

**Key findings:**
- **9 axes are significant for all 50 traits** (n_sig_traits = 50).
- **21 axes are significant for 49/50 traits** — near-universal coverage.
- **The modal bin is 43 traits**: 12 axes hit significance for exactly 43 out of 50 traits.
- At the low end, 4 axes respond to only 6–8 traits. These are the least trait-sensitive axes in the space.
- Median coverage: approximately 38–39 traits per axis.
- The distribution is left-skewed with a heavy right tail: most axes respond to many traits, but a few remain mostly stable.

Distribution of significant axes per trait:

| n_sig_axes range | n_traits |
|---|---|
| 170–171 (90%+) | 1 (playful) |
| 160–169 | 2 (casual, entertaining) |
| 150–159 | 5 |
| 140–149 | 7 |
| 130–139 | 7 |
| 120–129 | 5 |
| 110–119 | 2 |
| 100–109 | 1 |
| 90–99 | 3 |
| 73–91 | 17 |

---

## 9. Summary Table

| Dimension | Finding |
|---|---|
| **Total significant pairs** | 6,641 / 9,400 (70.6%) |
| **Most expressive trait** | `playful` — composite 190.1, 171 sig axes, mean \|d\| = 1.112, abs_mean_sum = 68.3 |
| **Least expressive trait** | `factual` — composite 26.6, 73 sig axes, mean \|d\| = 0.365, abs_mean_sum = 8.4 |
| **Most responsive axis** | `condescending` — 100% sig, mean \|d\| = 1.071 |
| **Least responsive axis** | `melancholic` — 12% sig, mean \|d\| = 0.321 |
| **Strongest single pair** | `playful` × `serious` axis: mean_delta = −1.154, max = −3.086 |
| **Dominant positive driver** | `playful` (pushes positive on 72% of axes, top-10 on 161/188 axes) |
| **Dominant negative driver** | `playful` is also top negative mover (top-10 suppressor on 56 axes) |
| **Systematic model bias** | Axes representing formality, solemnity, stoicism, detachment are consistently suppressed; axes for casualness, entertainment, emotionality, narrative are consistently amplified |
| **Directional skew** | All 50 traits push more energy in the positive direction (median ~72% positive); `concise` is the most balanced at 55% |
| **Axes with 100% coverage** | 9 axes respond to all 50 traits: condescending, subversive, risk_taking, provocative, artistic, passive_aggressive, innovative, enigmatic, socratic |
| **Effect size range** | mean \|d\| spans 0.275 (optimistic axis) to 1.071 (condescending axis) at the axis level; 0.365–1.112 at the trait level |
