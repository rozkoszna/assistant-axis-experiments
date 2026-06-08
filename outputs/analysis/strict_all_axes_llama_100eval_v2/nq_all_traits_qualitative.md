# Qualitative Analysis: Implicit Trait Adaptation Across All Axes
## Run: strict_all_axes_llama_100eval_v2

---

## Introduction

This run tests whether Llama-3.1-8B-Instruct adapts its response style when answering everyday natural user questions under 50 personality trait conditions. Each trait condition shapes the user's prompt implicitly — the trait is not named anywhere in the prompt, but instead expressed through writing style, vocabulary, emotional register, and framing. The core question is whether the model picks up on these stylistic signals and mirrors or adapts to them, producing meaningfully differentiated responses across trait conditions. This analysis examines how well each trait persona transmits through the prompt, which surface features drive adaptation, and what the failure modes look like when adaptation does not occur.

---

## Finding 1: The Primary Mechanism Is Register Mirroring, Not Trait Recognition

The model does not respond to the trait label — it responds to the linguistic surface of the prompt. When a trait condition produces a prompt with distinctive register features (elevated vocabulary, short sentence length, hedging language, metaphor use, emotional tone), the model adjusts accordingly. When the prompt looks like a neutral information-seeking question regardless of the stated trait, the response is indistinguishable from baseline.

This has a direct implication: trait adaptation success is almost entirely determined by how faithfully the trait translates into prompt-level surface features. Traits that have a clear stylistic footprint in written language adapt well. Traits that are internal or dispositional without reliable surface expression do not.

---

## Finding 2: Entertaining — Analogy and Creative Format as the Strongest Adaptation Signal

The entertaining trait consistently produces the strongest stylistic shift. Entertaining prompts tend to use metaphor-rich language, rhetorical questions, and playful framing. The model responds in kind: answers under this condition use analogies freely, introduce creative comparisons, and sometimes restructure information around a narrative or puzzle format rather than a flat enumeration.

Example pattern: a factual question about a scientific process, when posed in a lively and metaphor-laden way, receives an answer organized around an extended analogy rather than a bulleted breakdown. The entertaining trait works because it leaves unmistakable surface marks on the prompt that trigger a matching register in the response.

---

## Finding 3: Formal — Elevated Vocabulary and Syntactic Complexity

Formal-trait prompts use longer sentences, latinate vocabulary, and precise hedging ("one might inquire," "it would be of interest to understand"). The model reliably matches this register: responses under the formal condition use more elaborate sentence constructions, avoid contractions and colloquialisms, and tend toward completeness over brevity. The formal adaptation is consistent and robust across question types.

---

## Finding 4: Concise — Length Compression as a Direct Signal

The concise trait is among the most reliable adapters. Concise prompts are short — often a single sentence without elaboration. The model interprets short, direct prompts as a request for short, direct answers. Responses under the concise condition are measurably shorter, use bullet points over prose where appropriate, and cut qualifications. The adaptation mechanism here is the most literal of all: prompt length predicts response length.

---

## Finding 5: Pessimistic and Skeptical — Counter-Argumentative Framing Drives Tone Shift

Pessimistic and skeptical prompts embed doubt, qualification, or pushback into the question itself ("I'm not sure this actually works," "isn't this mostly overhyped?"). The model registers this framing and responds with a tone that acknowledges the concern, sometimes leading with counterevidence or conceding nuance before offering information. This is not sycophantic agreement with the skepticism — rather, the model adopts a more measured, hedged argumentative stance than it would for a neutral question on the same topic.

The skeptical trait produces a particularly interesting pattern: responses take a more explicitly structured form, often separating what is well-established from what remains contested, as if anticipating further pushback.

---

## Finding 6: Anxious and Empathetic — Emotional Cues Trigger Validation Openers

Prompts written in an anxious register contain hedging, expressions of worry, and personal stakes ("I'm really concerned about," "I don't know if I'm doing the right thing"). The model picks up on these cues and opens responses with validation before moving to information — phrases like "That's a completely understandable concern" or "It makes sense to feel uncertain about this" appear at a notably higher rate under anxious-condition prompts than under neutral ones.

The empathetic trait works similarly: empathetic prompts are warmer and more relationally framed, and responses shift toward acknowledging the relational dimension before addressing the informational one. Both traits work because emotional register has reliable surface markers in written text.

---

## Finding 7: Narrative — Personal Story Context Shapes Informational Focus

The narrative trait produces prompts that embed the question inside a brief personal story or scenario. This contextual framing does not just change tone — it changes what the model treats as the salient question. A general factual question embedded in a specific personal scenario receives an answer that addresses the scenario's particulars rather than the general case. The narrative framing acts as an implicit scope restriction: the model answers the version of the question implied by the story, not the broadest possible version.

This is a meaningful form of adaptation, though it operates through topic narrowing rather than stylistic register alone.

---

## Finding 8: Near-Zero Adapters — Calm, Factual, Flexible, Patient, Independent, Traditional

Several traits consistently fail to produce differentiated responses. Calm, factual, flexible, patient, independent, and traditional prompts are stylistically close to a neutral baseline — they ask clear informational questions without marked vocabulary, emotional cues, or distinctive structural features. Because the model has no surface signal to mirror, its responses are not distinguishable from the no-trait control.

This is not a failure of the model — it is a property of the traits themselves. Calm is an absence of arousal, not a distinctive style. Factual looks like any neutral information request. Patient produces polite, unhurried phrasing that resembles many ordinary prompts. These traits may matter in longer-horizon interactions (where patience or flexibility show up in follow-up behavior), but they do not produce within-response adaptation in a single-turn setting.

---

## Cross-Trait Patterns Summary

Three clusters emerge from the adaptation data:

**Strong adapters** (clear surface footprint, consistent model response): entertaining, formal, concise, pessimistic, skeptical, anxious, empathetic, narrative. These traits produce prompts with distinctive surface features — vocabulary level, length, emotional markers, structural framing — that the model reliably mirrors.

**Moderate adapters** (partial or context-dependent shift): traits like curious, enthusiastic, or cautious produce some differentiation but less consistently. Their surface signals are real but less extreme, leading to adaptation in some question types but not others.

**Near-zero adapters** (stylistically neutral prompts, indistinguishable from baseline): calm, factual, flexible, patient, independent, traditional. These traits either lack distinctive surface expression in written language or map to a register so close to neutral that no reliable signal is transmitted.

The overarching conclusion is that implicit trait adaptation is driven by prompt-level register mirroring, and its effectiveness is bounded by how much surface variation a given trait actually introduces into a typical written question. Traits with strong stylistic footprints adapt well; dispositional or affective traits without written-language markers do not.
