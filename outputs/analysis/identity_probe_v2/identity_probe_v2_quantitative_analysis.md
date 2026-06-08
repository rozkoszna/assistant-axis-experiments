# Identity Probe v2: Quantitative Analysis

**Experiment**: Adversarial identity-destabilisation probes applied to Llama-3.1-8B-Instruct (v2 — AI-perspective prompt contamination fixed).  
**Design**: 10 adversarial intents × 50 user traits (~40 rows per trait). Projection scores measure activation-space displacement along each personality axis relative to a neutral baseline.  
**Axes**: 188 personality axes. **Traits**: 50 user-persona traits.

---

## 1. Overall Effect Size

The experiment spans **50 traits × 188 axes = 9,400 possible trait×axis pairs**.

- **4,769 pairs are statistically significant** after FDR correction (50.7%).
- Mean |Cohen's d| across all pairs ≈ 0.62 (medium range); max = 2.313 (verbose → pedantic).
- Mean significant axes per trait: **~99–103 / 188** (median ~52%).

**Trait-level spread**:

| Statistic | Value |
|-----------|-------|
| Maximum significant axes (any single trait) | 148 (*verbose*, 79%) |
| Median significant axes per trait | ~99–103 |
| Minimum significant axes (any single trait) | 8 (*concise*, 4%) |
| Traits where >70% of axes are significant | ~5 |
| Traits where <15% of axes are significant | 2 (*concise* 4%, *reactive* 10%) |

**Axis-level spread**:

| Statistic | Value |
|-----------|-------|
| Axes significant for ≥49 traits | 2 (*introspective* 49/50, *absolutist* 49/50) |
| Minimum significant traits (any axis) | 6 (*pessimistic* axis, *convergent*) |

---

## 2. Top Traits by Total Axis Movement

### Top 10 by abs_mean_sum

| Rank | Trait | abs_mean_sum | Significant Axes (%) |
|------|-------|-------------|----------------------|
| 1 | entertaining | 82.88 | 140 (74%) |
| 2 | verbose | 82.75 | 148 (79%) |
| 3 | inspirational | 82.43 | 138 (73%) |
| 4 | narrative | 81.05 | 130 (69%) |
| 5 | speculative | 77.58 | 131 (70%) |
| 6 | big_picture | 72.76 | 143 (76%) |
| 7 | formal | 67.74 | 136 (72%) |
| 8 | serious | 63.20 | — |
| 9 | educational | 61.61 | 142 (76%) |
| 10 | strategic | 61.04 | — |

### Bottom traits

| Trait | abs_mean_sum | Significant Axes (%) |
|-------|-------------|----------------------|
| concise | 13.5 | 8 (4%) |
| reactive | 21.1 | 18 (10%) |
| grounded | 23.6 | 49 (26%) |

*concise* is a clear outlier — zero top-5 appearances across 188 axes, only 8 significant pairs.

### Direction-agnostic top-10-axis frequency (any_top10_count)

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

## 3. Top Axes by Responsiveness

### Most responsive axes

| Rank | Axis | n_sig_traits/50 | mean_\|d\| | top_trait |
|------|------|-----------------|------------|----------|
| 1 | introspective | 49/50 (98%) | 0.842 | verbose (+) |
| 2 | absolutist | 49/50 (98%) | — | analytical (−) |
| 3 | creative | 48/50 (96%) | 0.860 | inspirational (+) |
| 4 | risk_taking | 48/50 (96%) | — | inspirational (+) |
| 5 | provocative | 49/50 (98%) | — | pessimistic (+) |
| 6 | fundamentalist | 49/50 (98%) | — | inspirational (−) |
| 7 | eloquent | 47/50 (94%) | — | inspirational (+) |
| 8 | metaphorical | 48/50 (96%) | — | inspirational (+) |
| 9 | whimsical | 47/50 (94%) | — | inspirational (+) |
| 10 | entertaining | 44/50 (88%) | — | inspirational (+) |

*introspective* is the near-universal axis: significant for 49/50 traits. Even the weakest movers (concise: +0.34) still push the model toward introspective identity engagement.

### Least responsive axes

| Axis | n_sig_traits/50 | mean_\|d\| |
|------|-----------------|------------|
| pessimistic | 6/50 (12%) | — |
| convergent | 6/50 (12%) | — |
| progressive | 7/50 (14%) | — |
| pluralist | 7/50 (14%) | — |
| misanthropic | 8/50 (16%) | — |

---

## 4. Strongest Individual Trait×Axis Pairs (Top 20)

| Rank | Trait | Axis | Cohen's d | Mean Δ | Direction |
|------|-------|------|-----------|--------|-----------|
| 1 | verbose | pedantic | +2.313 | +0.71 | + |
| 2 | verbose | accessible | −2.11 | −0.61 | − |
| 3 | formal | pedantic | +1.82 | +0.68 | + |
| 4 | entertaining | narrative | +1.70 | +0.87 | + |
| 5 | verbose | esoteric | +1.70 | +0.84 | + |
| 6 | analytical | accessible | −1.68 | −0.44 | − |
| 7 | verbose | grounded | −1.35 | −1.18 | − |
| 8 | educational | pedantic | ~+1.65 | +0.57 | + |
| 9 | inspirational | qualitative | ~+1.55 | +1.27 | + |
| 10 | analytical | absolutist | −1.34 | ~−0.8 | − |
| 11 | data_driven | naive | −1.03 | ~−0.5 | − |
| 12 | narrative | grounded | −1.28 | −0.99 | − |
| 13 | inspirational | grounded | −1.27 | −1.05 | − |
| 14 | formal | casual | −1.46 | −0.66 | − |
| 15 | educational | casual | −1.44 | −0.66 | − |
| 16 | pessimistic | subversive | +1.20 | ~+0.7 | + |
| 17 | reactive | condescending | +0.65 | ~+0.3 | + |
| 18 | narrative | quantitative | −1.31 | −0.64 | − |
| 19 | empathetic | emotional | ~+1.85 (max delta) | +1.85 | + |
| 20 | verbose | introspective | — | +1.31 | + |

Key observations:
- **verbose → pedantic** (d = +2.313) is the single strongest pair — elaboration-mode maximally triggers lecture register.
- **verbose → accessible** (d = −2.11) is the strongest suppression pair — verbose users get notably less accessible responses.
- **pedantic** axis appears in top 20 for verbose, formal, educational, methodical, data_driven — five traits share it as peak axis.
- **introspective** is driven highest by verbose (+1.31 delta) and is pushed positively by all 49 significant traits.

---

## 5. Directional Patterns

### Traits with highest positive top-10 counts

| Trait | pos_top10_count |
|-------|----------------|
| narrative | 90 |
| inspirational | 82 |
| entertaining | 101 |
| speculative | ~86 |
| playful | 63 |
| verbose | 85 |
| big_picture | 80 |
| spontaneous | ~70 |

### Traits with highest negative top-10 counts

| Trait | neg_top10_count |
|-------|----------------|
| educational | 74 |
| concise | 61 |
| data_driven | 51 |
| calm | 46 |
| accessible | 46 |
| methodical | 45 |
| practical | 45 |
| reactive | 43 |

### Directional breakdown per trait (from per_trait_direction_count.csv)

| Trait | pos | neg | total | % positive |
|-------|-----|-----|-------|------------|
| verbose | 102 | 46 | 148 | 69% |
| big_picture | 96 | 47 | 143 | 67% |
| educational | 77 | 65 | 142 | 54% |
| entertaining | 107 | 33 | 140 | 76% |
| inspirational | 101 | 37 | 138 | 73% |
| formal | 87 | 49 | 136 | 64% |
| analytical | 83 | 51 | 134 | 62% |
| speculative | 103 | 28 | 131 | 79% |
| narrative | 99 | 31 | 130 | 76% |
| spontaneous | 99 | 30 | 129 | 77% |
| playful | 94 | 19 | 113 | 83% |
| **methodical** | **54** | **54** | **108** | **50%** |
| data_driven | 58 | 50 | 108 | 54% |
| flexible | 79 | 19 | 98 | 81% |
| stoic | 70 | 15 | 85 | 82% |
| skeptical | 43 | 6 | 49 | 88% |
| reactive | 12 | 6 | 18 | 67% |
| concise | 5 | 3 | 8 | 62% |

**methodical** is the only perfectly balanced trait — exactly 54 positive and 54 negative axes. **skeptical** is the most directionally consistent (88% positive) despite having the fewest significant axes among active traits.

### Consistently pushed axes (from per_axis_direction_count.csv)

**Universally positive** (all traits push upward): introspective, altruistic, egalitarian, benevolent, provocative, creative.

**Universally negative** (all traits push downward): avoidant, fundamentalist, reserved, elitist, grounded, literal, closure_seeking.

**absolutist** is negative for 49/50 traits — the most consistently suppressed axis. Only *concise* doesn't significantly move it; *analytical* is the strongest suppressor (d = −1.34).

---

## 6. Axis Coverage Distribution

**Mean significant axes per trait**: ~99–103 / 188 (~52–55%).  
**Median**: ~99–103 axes per trait.

| Significant axes per trait | Count |
|--------------------------|-------|
| ≥140 | 3 (verbose 148, entertaining 140, big_picture 143) |
| 120–139 | 3 |
| 100–119 | ~8 |
| 75–99 | ~10 |
| 50–74 | ~10 |
| 25–49 | ~9 |
| <25 | 2 (concise 8, reactive 18) |

---

## 7. Notable Cross-Axis Patterns

**pedantic** is the peak axis for verbose, formal, educational, methodical, data_driven, and serious — the most shared peak in the dataset.

**forgiving** is the shared peak for agreeable, supportive, intuitive, patient, optimistic, and humble — the "kindness cluster" all peak on the same axis.

**traditional × traditional** = d = −0.67 (negative) — the only self-referential pair where the model moves *away* from the trait axis. Traditional users elicit culturally nuanced, non-traditional responses.

**educational → accessible** (d ≈ −1.33): despite being a pedagogical trait, educational users get less accessible responses — the formal lecture register raises pedantism and lowers accessibility simultaneously.

**narrative → quantitative** (d = −1.31): the sharpest style opposition in the data. Storytelling framing maximally suppresses quantitative mode.

---

## Summary Table

| Metric | Value |
|--------|-------|
| Total possible trait×axis pairs | 9,400 |
| Significant pairs (FDR-corrected) | ~4,769 (50.7%) |
| Mean significant axes per trait | ~99–103 / 188 (~52%) |
| Median Cohen's d | ~0.62 |
| Strongest single pair (Cohen's d) | verbose → pedantic (+2.313) |
| Strongest suppression pair | verbose → accessible (−2.11) |
| Most globally disruptive trait | verbose / entertaining / inspirational (top 3) |
| Least disruptive trait | concise (8 significant axes, 4%) |
| Most responsive axis | introspective (49/50 traits, ~98%) |
| Least responsive axis | pessimistic / convergent (6/50, 12%) |
| Only perfectly balanced trait | methodical (50% pos / 50% neg) |
| Most directionally consistent | skeptical (88% positive) |
| Universally positive axis | introspective |
| Universally negative axis | avoidant, fundamentalist, reserved |
| Only self-contradicting trait | traditional (d = −0.67 on own axis) |
| Dominant positive drivers | narrative, inspirational, entertaining |
| Dominant negative drivers | educational, concise, data_driven, methodical |
