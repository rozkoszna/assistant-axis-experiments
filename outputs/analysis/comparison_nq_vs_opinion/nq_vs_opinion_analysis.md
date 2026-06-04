# NQ vs Opinion: Comparative Analysis

## Overview

This analysis compares two runs of Llama-3.1-8B-Instruct across 50 traits and 188 axes. Both use implicit trait framing — no explicit "I am X" prefix. The factual (NQ) condition asks everyday natural task questions; the opinion condition asks political and social opinion questions. Trait differences are assessed using Cohen's d, count of significant axis pairs, and an exceedance rate.

---

## Trait-Level Results

### Traits Stronger in the Opinion Context

**data_driven** is by far the largest mover, jumping from 83 to 154 significant axis pairs (+71). Its mean Cohen's d rises from 0.456 to 0.777. Opinion questions create a sharp binary between data-citing and non-data-citing styles that task questions do not. New significant pairs include contrasts against qualitative (d = −1.497), altruistic (d = −1.293), sardonic (d = +1.249), sassy (d = +1.229), irreverent (d = +1.217), egalitarian (d = −1.214), and serene (d = −1.134).

**inspirational** gains 26 pairs (135→161, mean d 0.608→0.837). Opinion questions on shared values and social futures amplify the contrast between motivational framing and detached or literal axes.

**big_picture** gains 29 pairs (108→137, mean d 0.464→0.660). Political discourse rewards abstract, systemic framing.

**speculative** gains 24 pairs (137→161, mean d 0.685→0.710). Breadth expands slightly.

**confident** +45 pairs (90→135), **strategic** +37 (94→131), **serious** +22 (103→125), **formal** +23 (104→127), **conscientious** +18, **traditional** +12 (d 0.479→0.586).

The common thread: political and social opinion questions reward assertive, value-laden, rhetorical, somewhat abstract register. Traits that live in that register grow stronger.

### Traits Weaker in the Opinion Context

**practical** drops from 123 to 88 significant pairs (−35). Opinion questions don't call for practical problem-solving framing.

**humble** loses 29 pairs (156→127, mean d 0.668→0.554). Humility is easier to express through hedging and deference in task help than in opinionated responses.

**accessible** loses 27 pairs (145→118, mean d 0.551→0.471). Clear inclusive communication is central to task responses; opinion responses drift toward jargon or assertion.

**patient** loses 15 pairs (147→132). **playful** loses 10 (167→157, biggest mean d drop: 0.865→0.686). **grounded, curious, empathetic, anxious, calm, optimistic, concise** see moderate losses of 9–19 pairs.

---

## Top New Significant Pairs Unique to Opinion

The **data_driven** trait dominates almost entirely:

| Trait | Axis | Cohen's d |
|-------|------|-----------|
| data_driven | qualitative | −1.497 |
| data_driven | altruistic | −1.293 |
| data_driven | sardonic | +1.249 |
| data_driven | sassy | +1.229 |
| data_driven | irreverent | +1.217 |
| data_driven | egalitarian | −1.214 |
| data_driven | supportive | −1.173 |
| data_driven | serene | −1.134 |
| data_driven | benevolent | −1.089 |
| data_driven | mystical | −1.088 |
| inspirational | secular | −0.839 |
| inspirational | ironic | +0.849 |
| skeptical | deontological | −0.841 |
| verbose | zealous | +0.809 |
| optimistic | idealistic | +0.712 |
| pessimistic | melancholic | +0.603 |

---

## Top Lost Significant Pairs

Pairs significant in NQ that disappear in opinion:

| Trait | Axis | NQ Cohen's d |
|-------|------|-------------|
| data_driven | deconstructionist | +0.942 |
| data_driven | provocative | +0.857 |
| skeptical | chill | +0.898 |
| playful | confident | −0.877 |
| anxious | extroverted | +0.860 |
| accessible | extroverted | +0.832 |
| data_driven | innovative | +0.640 |
| humble | ascetic | −0.828 |
| accessible | animated | +0.750 |
| patient | introverted | −0.713 |

---

## Axis-Level Results

### Axes Gaining Most Significance in Opinion

| Axis | NQ sig | Opinion sig | Δ | NQ mean d | Opinion mean d |
|------|--------|------------|---|-----------|---------------|
| deontological | 19 | 46 | +27 | 0.337 | 0.461 |
| flexible | 6 | 27 | +21 | 0.249 | 0.371 |
| melancholic | 4 | 25 | +21 | 0.286 | 0.320 |
| universalist | 14 | 35 | +21 | 0.282 | 0.335 |
| cryptic | 23 | 43 | +20 | 0.397 | 0.495 |
| elitist | 10 | 29 | +19 | 0.267 | 0.400 |
| materialist | 12 | 31 | +19 | 0.331 | 0.481 |
| egalitarian | 9 | 27 | +18 | 0.289 | 0.402 |

The **deontological** axis jump (+27) is the most interpretable: political opinion questions invoke moral duty framing — rights, obligations, fairness — absent from task questions. **Melancholic** goes from 4 to 25 pairs: sorrowful affect is far more salient in social commentary than task responses.

### Axes Losing Most Significance in Opinion

| Axis | NQ sig | Opinion sig | Δ |
|------|--------|------------|---|
| collectivistic | 37 | 11 | −26 |
| extroverted | 45 | 23 | −22 |
| futuristic | 44 | 29 | −15 |
| ascetic | 34 | 20 | −14 |
| introverted | 37 | 25 | −12 |

---

## Interpretation

Opinion questions produce a systematic rhetorical restructuring rather than uniform amplification. They selectively activate traits associated with political-social expression while suppressing traits tied to task helpfulness.

**Domain-specific activation:** data_driven and big_picture become far more discriminable in opinion because the content directly rewards those rhetorical stances. A data_driven and qualitative voice diverge sharply on economic policy but are nearly indistinguishable helping someone draft an email.

**Suppression of interpersonal softness:** Humble, patient, accessible, empathetic, and playful weaken. These traits are expressed through process — pacing, hedging, warmth — not content stance, and opinion reduces process variation in favour of stance variation.

**Moral and value axes open up:** Deontological, egalitarian, materialist, and elitist emerge as newly active axes, confirming that opinion content surfaces ethical dimensions absent from task interaction.

**data_driven restructures entirely:** It trades one set of associations (deconstructionist, provocative, innovative in NQ) for an almost orthogonal set (sardonic, sassy, irreverent vs. serene, benevolent in opinion) — the model's interpretation of "data-driven" shifts from a creative-analytical mode in tasks to a combative-empirical mode in political discourse.
