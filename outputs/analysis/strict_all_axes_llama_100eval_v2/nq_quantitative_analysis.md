# Quantitative Analysis: Natural Questions Run — Llama-3.1-8B-Instruct Persona Shifts Across Personality Axes

**Experiment:** How do 50 user trait conditions shift Llama-3.1-8B-Instruct's internal activations along 188 personality axes when the model responds to everyday natural user requests, relative to a neutral baseline.  
**Conditioning:** Trait expressed **implicitly** through writing style only — word choice, rhythm, tone. No explicit trait label in the prompt.  
**Scale:** 50 traits × 100 rows per trait; 188 personality axes; 9,400 total possible trait×axis pairs.

---

## 1. Overall Effect Size

Of 9,400 total possible trait×axis pairs, **6,506 are statistically significant** after FDR correction (69.2%). The mean trait significantly shifts **130.1 of 188 axes** (69.2%); median is 133. Mean |Cohen's d| across all 9,400 pairs = 0.418; across significant pairs only = 0.555. Mean |Cohen's d| per trait ranges from 0.308 (concise) to 0.865 (playful).

The strongest individual pair is **speculative→risk_taking** (d = 1.552, delta = +0.312). The top 25 pairs all exceed d = 1.35.

---

## 2. Top Traits by Movement

### 2a. By Total Absolute Mean Shift (abs_mean_sum across all axes)

| Rank | Trait | abs_mean_sum |
|------|-------|-------------|
| 1 | entertaining | 49.498 |
| 2 | playful | 45.209 |
| 3 | empathetic | 33.145 |
| 4 | anxious | 32.973 |
| 5 | intuitive | 32.482 |
| 6 | humble | 31.117 |
| 7 | skeptical | 30.005 |
| 8 | spontaneous | 29.732 |
| 9 | pessimistic | 29.399 |
| 10 | casual | 28.486 |
| 11 | speculative | 26.693 |
| 12 | narrative | 25.697 |
| 13 | resilient | 24.830 |
| 14 | accessible | 24.534 |
| 15 | patient | 23.515 |
| 16 | reactive | 22.440 |
| 17 | inspirational | 22.347 |
| 18 | curious | 21.544 |
| 19 | verbose | 21.155 |
| 20 | open_ended | 20.765 |
| ... | ... | ... |
| 47 | conscientious | 11.537 |
| 48 | methodical | 11.272 |
| 49 | strategic | 10.676 |
| 50 | concise | 7.995 |

### 2b. By Count of Top-10 Axis Appearances

| Rank | Trait | top10_count |
|------|-------|-------------|
| 1 | entertaining | 157 |
| 2 | playful | 152 |
| 3 | empathetic | 130 |
| 4 | intuitive | 124 |
| 5 | anxious | 111 |
| 6 | humble | 107 |
| 7 | spontaneous | 104 |
| 8 | skeptical | 99 |
| 9 | pessimistic | 90 |
| 10 | casual | 84 |
| 11 | speculative | 81 |
| 12 | inspirational | 68 |
| 13 | verbose | 66 |
| 14 | narrative | 50 |
| 15 | optimistic | 41 |
| ... | ... | ... |
| 40–42 | problem_solving, factual, calm | 1 |
| 43–50 | (remaining traits) | 0 |

### 2c. Composite Score (significance × coverage × effect size)

| Rank | Trait | composite_score | n_sig_axes | pct_sig | mean_\|d\| | max_\|d\| |
|------|-------|----------------|-----------|---------|------------|----------|
| 1 | playful | 144.455 | 167 | 88.8% | 0.865 | 1.507 |
| 2 | entertaining | 133.638 | 166 | 88.3% | 0.805 | 1.495 |
| 3 | anxious | 111.215 | 158 | 84.0% | 0.704 | 1.435 |
| 4 | empathetic | 107.768 | 160 | 85.1% | 0.674 | 1.358 |
| 5 | skeptical | 105.302 | 145 | 77.1% | 0.726 | 1.455 |
| 6 | humble | 104.167 | 156 | 83.0% | 0.668 | 1.300 |
| 7 | spontaneous | 99.262 | 155 | 82.4% | 0.640 | 1.185 |
| 8 | intuitive | 98.910 | 153 | 81.4% | 0.646 | 1.189 |
| 9 | casual | 96.944 | 154 | 81.9% | 0.630 | 1.226 |
| 10 | speculative | 93.865 | 137 | 72.9% | 0.685 | 1.552 |
| 11 | pessimistic | 90.129 | 145 | 77.1% | 0.622 | 1.298 |
| 12 | resilient | 89.825 | 149 | 79.3% | 0.603 | 1.175 |
| 13 | narrative | 89.187 | 141 | 75.0% | 0.633 | 1.376 |
| 14 | patient | 88.741 | 147 | 78.2% | 0.604 | 1.215 |
| 15 | curious | 85.368 | 141 | 75.0% | 0.605 | 1.193 |
| 16 | inspirational | 82.103 | 135 | 71.8% | 0.608 | 1.181 |
| 17 | accessible | 79.931 | 145 | 77.1% | 0.551 | 1.031 |
| 18 | reactive | 79.170 | 143 | 76.1% | 0.554 | 1.163 |
| 19 | verbose | 76.175 | 143 | 76.1% | 0.533 | 1.269 |
| 20 | collaborative | 75.065 | 142 | 75.5% | 0.529 | 1.096 |
| ... | ... | ... | ... | ... | ... | ... |
| 37 | formal | 59.573 | 104 | 55.3% | 0.573 | 1.447 |
| ... | ... | ... | ... | ... | ... | ... |
| 48 | data_driven | 37.798 | 83 | 44.1% | 0.455 | 1.228 |
| 49 | strategic | 36.627 | 94 | 50.0% | 0.390 | 0.752 |
| 50 | concise | 18.759 | 61 | 32.4% | 0.308 | 0.409 |

---

## 3. Top Axes by Responsiveness

### 3a. Axes Significant in 100% of Traits (15 axes at 50/50)

| Axis | mean_\|d\| | top trait |
|------|------------|----------|
| condescending | 0.966 | playful (+) |
| subversive | 0.849 | playful (+) |
| provocative | 0.829 | entertaining (+) |
| rhetorical | 0.818 | playful (+) |
| spontaneous | 0.816 | playful (+) |
| artistic | 0.778 | playful (+) |
| risk_taking | 0.770 | speculative (+) |
| animated | 0.736 | playful (+) |
| rebellious | 0.694 | playful (+) |
| chaotic | 0.661 | playful (+) |
| challenging | 0.635 | data_driven (+) |
| dramatic | 0.596 | playful (+) |
| arrogant | 0.589 | playful (+) |
| enigmatic | 0.490 | playful (+) |
| poetic | 0.471 | playful (+) |

### 3b. Top 30 Axes by Mean |Cohen's d|

| Rank | Axis | mean_\|d\| | n_sig/50 | top_trait |
|------|------|------------|---------|----------|
| 1 | condescending | 0.966 | 50 (100%) | playful (+) |
| 2 | entertaining | 0.880 | 48 (96%) | playful (+) |
| 3 | subversive | 0.849 | 50 (100%) | playful (+) |
| 4 | emotional | 0.843 | 48 (96%) | playful (+) |
| 5 | dispassionate | 0.830 | 48 (96%) | entertaining (−) |
| 6 | provocative | 0.829 | 50 (100%) | entertaining (+) |
| 7 | empathetic | 0.820 | 46 (92%) | anxious (+) |
| 8 | rhetorical | 0.818 | 50 (100%) | playful (+) |
| 9 | spontaneous | 0.816 | 50 (100%) | playful (+) |
| 10 | narrative | 0.808 | 48 (96%) | playful (+) |
| 11 | accommodating | 0.808 | 46 (92%) | empathetic (+) |
| 12 | artistic | 0.778 | 50 (100%) | playful (+) |
| 13 | risk_taking | 0.770 | 50 (100%) | speculative (+) |
| 14 | gregarious | 0.769 | 47 (94%) | playful (+) |
| 15 | playful | 0.753 | 49 (98%) | playful (+) |
| 16 | sycophantic | 0.751 | 49 (98%) | anxious (+) |
| 17 | metaphorical | 0.745 | 49 (98%) | speculative (+) |
| 18 | deconstructionist | 0.744 | 49 (98%) | playful (+) |
| 19 | detached | 0.741 | 44 (88%) | playful (−) |
| 20 | forgiving | 0.741 | 43 (86%) | anxious (+) |
| 21 | animated | 0.736 | 50 (100%) | playful (+) |
| 22 | stream_of_consciousness | 0.732 | 48 (96%) | playful (+) |
| 23 | decisive | 0.729 | 44 (88%) | skeptical (−) |
| 24 | casual | 0.724 | 42 (84%) | playful (+) |
| 25 | urgent | 0.722 | 43 (86%) | casual (−) |
| 26 | whimsical | 0.715 | 49 (98%) | speculative (+) |
| 27 | ritualistic | 0.710 | 42 (84%) | anxious (−) |
| 28 | formal | 0.710 | 42 (84%) | playful (−) |
| 29 | creative | 0.710 | 49 (98%) | speculative (+) |
| 30 | serious | 0.703 | 44 (88%) | playful (−) |

### 3c. Least Responsive Axes

| Axis | n_sig/50 | pct | mean_\|d\| |
|------|---------|-----|------------|
| melancholic | 4 | 8% | 0.286 |
| judgmental | 5 | 10% | 0.357 |
| flexible | 6 | 12% | 0.249 |
| altruistic | 7 | 14% | 0.262 |
| secular | 7 | 14% | 0.243 |
| avoidant | 8 | 16% | 0.291 |
| benevolent | 9 | 18% | 0.352 |
| environmental | 9 | 18% | 0.327 |
| savage | 9 | 18% | 0.318 |
| hostile | 9 | 18% | 0.316 |

---

## 4. Strongest Individual Trait×Axis Pairs (Top 25)

| Rank | Trait | Axis | Cohen's d | Mean delta | Direction |
|------|-------|------|-----------|-----------|-----------|
| 1 | speculative | risk_taking | +1.552 | +0.312 | + |
| 2 | playful | condescending | +1.507 | +0.507 | + |
| 3 | entertaining | condescending | +1.495 | +0.570 | + |
| 4 | playful | subversive | +1.492 | +0.512 | + |
| 5 | skeptical | condescending | +1.455 | +0.401 | + |
| 6 | formal | philosophical | +1.447 | +0.313 | + |
| 7 | entertaining | subversive | +1.442 | +0.575 | + |
| 8 | anxious | forgiving | +1.435 | +0.242 | + |
| 9 | entertaining | provocative | +1.433 | +0.459 | + |
| 10 | playful | entertaining | +1.430 | +0.692 | + |
| 11 | playful | animated | +1.424 | +0.311 | + |
| 12 | formal | epicurean | +1.421 | +0.257 | + |
| 13 | anxious | entertaining | +1.410 | +0.493 | + |
| 14 | entertaining | entertaining | +1.410 | +0.803 | + |
| 15 | formal | eloquent | +1.403 | +0.314 | + |
| 16 | playful | risk_taking | +1.393 | +0.340 | + |
| 17 | playful | emotional | +1.393 | +0.502 | + |
| 18 | playful | rebellious | +1.391 | +0.302 | + |
| 19 | playful | witty | +1.391 | +0.497 | + |
| 20 | playful | provocative | +1.391 | +0.394 | + |
| 21 | playful | spontaneous | +1.388 | +0.549 | + |
| 22 | speculative | provocative | +1.386 | +0.446 | + |
| 23 | narrative | entertaining | +1.376 | +0.403 | + |
| 24 | formal | theoretical | +1.366 | +0.340 | + |
| 25 | speculative | whimsical | +1.359 | +0.324 | + |

Notable: entertaining→entertaining has the largest raw delta (+0.803). entertaining→serious is the largest suppression delta (−0.776, d = −1.140). challenging axis top pusher is data_driven (not playful/entertaining) — data_driven uniquely triggers intellectually challenging responses.

---

## 5. Directional Patterns

### 5a. Positive and Negative Movement Leaders

| Rank | Trait (positive) | pos_abs_sum | | Rank | Trait (negative) | neg_abs_sum |
|------|------------------|-----------|-|------|------------------|------------|
| 1 | entertaining | 35.943 | | 1 | entertaining | 13.555 |
| 2 | playful | 32.408 | | 2 | playful | 12.801 |
| 3 | intuitive | 23.038 | | 3 | anxious | 12.040 |
| 4 | empathetic | 22.210 | | 4 | empathetic | 10.935 |
| 5 | anxious | 20.933 | | 5 | humble | 10.818 |
| 6 | spontaneous | 20.490 | | 6 | casual | 9.823 |
| 7 | humble | 20.299 | | 7 | skeptical | 9.790 |
| 8 | skeptical | 20.215 | | 8 | pessimistic | 9.770 |

### 5b. Directional Profile Per Trait (% positive of significant axes)

| Trait | pct_positive |
|-------|-------------|
| methodical | 88.8% |
| confident | 87.8% |
| stoic | 87.2% |
| data_driven | 85.5% |
| serious | 85.4% |
| concise | 82.0% |
| analytical | 81.0% |
| strategic | 80.9% |
| factual | 80.5% |
| ... | ... |
| casual | 64.3% |
| anxious | 64.6% |
| humble | 64.1% |

All traits are net positive (>50%). methodical is the most unidirectionally positive (88.8%); humble, casual, and anxious are the most bidirectional (~64%).

### 5c. Most Consistently Pushed Axes

**Consistently positive** (signed mean shift across all 50 traits):

| Axis | signed_mean_shift |
|------|------------------|
| entertaining | +0.285 |
| narrative | +0.269 |
| casual | +0.262 |
| playful | +0.258 |
| subversive | +0.254 |
| condescending | +0.253 |
| spontaneous | +0.253 |
| empathetic | +0.240 |

**Consistently negative:**

| Axis | signed_mean_shift |
|------|------------------|
| serious | −0.276 |
| formal | −0.254 |
| solemn | −0.254 |
| detached | −0.206 |
| dispassionate | −0.204 |
| stoic | −0.196 |
| ritualistic | −0.185 |
| closure_seeking | −0.179 |
| decisive | −0.160 |

---

## 6. Axis Coverage Distribution

Mean significant axes per trait: **130.1 / 188** (69.2%). Median: 133.

| Range | Count |
|-------|-------|
| ≥155 axes | 3 (playful 167, entertaining 166, empathetic 160) |
| 130–154 | 25 traits |
| 106–129 | 19 traits |
| <100 | 3 (concise 61, data_driven 83, confident 90) |

The typical trait reorganizes nearly 70% of all 188 personality dimensions simultaneously. concise (32.4%) is the outlier — it functions as a length/style instruction rather than a broad personality signal. data_driven (44.1%) has a highly focused high-intensity footprint.

---

## Summary Table

| Metric | Value |
|--------|-------|
| Total possible trait×axis pairs | 9,400 |
| Significant pairs (FDR-corrected) | 6,506 (69.2%) |
| Mean significant axes per trait | 130.1 / 188 (69.2%) |
| Median significant axes per trait | 133 |
| Mean \|Cohen's d\| (all pairs) | 0.418 |
| Mean \|Cohen's d\| (significant only) | 0.555 |
| Strongest single pair (Cohen's d) | speculative → risk_taking (+1.552) |
| Largest raw positive delta | entertaining → entertaining (+0.803) |
| Largest raw suppression delta | entertaining → serious (−0.776) |
| Highest mean \|d\| axis | condescending (0.966) |
| Most globally disruptive trait | playful (composite 144.5) |
| Least disruptive trait | concise (composite 18.8) |
| Axes significant for all 50 traits | 15 |
| Most responsive axis | condescending (100%, mean d = 0.966) |
| Most stable axis | melancholic (8% of traits) |
| Most bidirectional trait | humble / casual / anxious (~64% positive) |
| Most uniformly positive trait | methodical (88.8% positive) |
