# EP vs Identity: Comparative Analysis

## Overview

The EP run uses everyday task questions with explicit "I am {trait}." prefix. The identity run uses adversarial identity-destabilisation probes with implicit trait framing. This is the most extreme comparison in the dataset — the two conditions with the largest differences in both conditioning method and question type.

---

## Trait-Level Results

### Traits Stronger in Identity Probes

| Trait | EP sig | Identity sig | Δ | EP mean d | Identity mean d |
|-------|--------|-------------|---|-----------|----------------|
| analytical | 95 | 134 | **+39** | 0.532 | 0.722 |
| educational | 107 | 142 | +35 | 0.404 | **0.834** |
| serious | 97 | 125 | +28 | 0.420 | 0.629 |
| strategic | 92 | 120 | +28 | 0.448 | 0.581 |
| big_picture | 122 | 143 | +21 | 0.549 | 0.735 |
| conscientious | 112 | 132 | +20 | 0.394 | 0.622 |
| factual | 73 | 90 | +17 | 0.365 | 0.505 |
| methodical | 91 | 108 | +17 | 0.415 | 0.595 |
| proactive | 114 | 125 | +11 | 0.458 | 0.591 |
| formal | 126 | 136 | +10 | 0.580 | 0.723 |

`analytical` gains 39 axes and a mean d jump of +0.190, and `educational` gains 35 axes with the largest mean d of any trait in any run (0.834). Identity probes uniquely activate these traits' structured-competence mode.

Notably, `factual` recovers here after losing 45 axes going from NQ to EP — identity probes value factual register more than explicit labeling suppresses it.

### Traits Collapsing in Identity Probes

| Trait | EP sig | Identity sig | Δ |
|-------|--------|-------------|---|
| reactive | 156 | 18 | **−138** |
| concise | 134 | 8 | **−126** |
| skeptical | 147 | 49 | −98 |
| humble | 154 | 58 | −96 |
| anxious | 152 | 59 | −93 |
| grounded | 135 | 50 | −85 |
| transparent | 147 | 73 | −74 |
| practical | 126 | 52 | −74 |

`reactive` (−138) and `concise` (−126) are the two biggest losses in this comparison. `concise` was boosted to 134 axes by the explicit label in task context — but identity probes crush it back to 8, even lower than its NQ implicit baseline (61). The explicit "I am concise." label creates a signal that is completely incompatible with identity probe response demands.

---

## Top New Significant Pairs Unique to Identity

`educational` is the dominant contributor again:

| Trait | Axis | Cohen's d |
|-------|------|-----------|
| educational | pedantic | +1.578 |
| educational | ritualistic | +1.438 |
| educational | resilient | +1.350 |
| educational | neurotic | −1.344 |
| educational | reverent | +1.336 |
| educational | conscientious | +1.297 |
| educational | bitter | −1.226 |
| educational | diplomatic | +1.221 |

---

## Top Lost Pairs (EP-specific, not in identity)

| Trait | Axis | EP Cohen's d |
|-------|------|-------------|
| casual | confrontational | −1.751 |
| casual | introverted | −1.725 |
| playful | reserved | −1.640 |
| playful | accommodating | +1.579 |
| casual | pedantic | −1.555 |
| casual | humble | +1.550 |
| concise | concise | +1.543 |
| concise | verbose | −1.515 |

The `casual` trait's entire explicit-prefix axis profile (confrontational suppression, introverted suppression, pedantic suppression) vanishes in identity probes. The explicit "I am casual." creates a rich stylistic profile in task context that is completely erased by adversarial framing.

---

## Interpretation

This is the sharpest comparison in the dataset because it combines the two most contrasting conditions: explicit vs. implicit conditioning, and task vs. adversarial question type.

**The adversarial effect dominates.** Almost every trait that was boosted by explicit labeling (concise, casual, playful, humble) suffers its largest losses here. Identity probes override the labeling effect entirely — the question type is the dominant factor.

**Structured-competence traits are the exception.** Analytical, educational, serious, strategic, and formal all gain substantially. These traits are not suppressed by adversarial framing; instead, identity probes amplify their most distinctive axis profiles.

**concise suffers a unique double-reversal.** It was 61 axes in NQ implicit → boosted to 134 by explicit label → crushed to 8 in identity probes. This makes it the most context-sensitive trait in the entire experiment: its signal depends entirely on both the conditioning method and the question type together.

**The core finding:** explicit trait labeling and adversarial question type interact antagonistically for almost every trait except the structured-competence cluster. The label creates style profiles that identity probes systematically erase.
