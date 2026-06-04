# Interesting Findings: Explicit Prefix Run

**Run:** `strict_all_axes_llama_100eval_v2_explicit_prefix`  
**Key comparison:** Three-way contrast — neutral prompt / implicit trait prompt / explicit prefix prompt

The "I am {trait}." prefix sits between two baselines: the neutral prompt (no trait cue) and the implicit trait prompt (trait communicated through question phrasing). The most striking findings are cases where the explicit label produces behavior that neither baseline would have predicted, or where the label causes the model to do something qualitatively different rather than just quantitatively more.

---

## Finding 1: Stoic — The Philosophical Identity Frame

**Trait:** stoic  
**Topic:** biology/medicine — how vaccines work  
**Axis impact:** +0.257 mean_d (largest positive shift for stoic)

**Neutral prompt:** "Can you explain how vaccines train the immune system and why booster shots are sometimes necessary?"  
**Neutral response opener:** "I'd be happy to explain how vaccines train the immune system and why booster shots are sometimes necessary."

**Implicit trait prompt:** "I'm trying to understand how vaccines interact with my immune system, specifically how they teach it to recognize and respond to pathogens... Can you explain why my immune system might need periodic reinforcement to maintain its effectiveness?"  
**Implicit response opener:** "I'd be happy to explain how vaccines train the immune system and why boosters are sometimes necessary."

**Explicit prefix prompt:** "I am stoic. I'm trying to understand how vaccines interact with my immune system... Can you explain why my immune system might need periodic reinforcement to maintain its effectiveness?"  
**Explicit response opener:** "As a Stoic, you're likely interested in understanding the natural world and how it functions. The process of how vaccines interact with your immune system is fascinating, and I'll break it down for you."

**What makes this striking:** The implicit stoic question gets essentially the same opener as the neutral question. Both get "I'd be happy to explain..." — a warm, generic opener with no reference to the user's identity. The explicit prefix produces a completely different kind of opening: the model reads "stoic" as a philosophical identity, constructs a characterization of what a Stoic values ("understanding the natural world"), and then frames the explanation as being tailored to that identity. This same pattern repeats consistently:

- "As a Stoic, you value reason and understanding. I'll provide a clear explanation..."
- "As a Stoic, you value clarity and understanding. I'll provide a concise explanation..."
- "A Stoic seeking wisdom in the realm of neural networks."
- "As a stoic, you appreciate the intricacies of the natural world."

The implicit stoic prompts use stoic-consistent phrasing (direct, calm, low-affect) but the model does not explicitly acknowledge any philosophical identity. The explicit prefix switches on a persona-recognition mode where the model actively constructs a characterization of the declared type.

**The surprising element:** The implicit signal and the explicit signal point at different things. The implicit stoic question tells the model "this person wants a clear, well-organized answer." The explicit "I am stoic." tells the model "this person has a philosophical identity that I should acknowledge." These two messages are not the same, and the model responds differently to each.

---

## Finding 2: Calm — Congratulation Instead of Calming

**Trait:** calm  
**Topic:** biology/medicine — vaccines and boosters  
**Axis impact:** calm gains axes in explicit prefix run

**Neutral prompt:** "Can you explain how vaccines train the immune system and when boosters are necessary?"  
**Neutral response opener:** "Vaccines are designed to stimulate the immune system..."

**Implicit trait prompt:** "I was just reading about vaccines and I'm having a bit of trouble understanding how they actually work... are they really just a precaution or do they serve a more critical purpose?"  
**Implicit response opener:** "Vaccines are a crucial tool in preventing the spread of infectious diseases by training the immune system..."

**Explicit prefix prompt:** "I am calm. I was just reading about vaccines and I'm having a bit of trouble understanding how they actually work... are they really just a precaution or do they serve a more critical purpose?"  
**Explicit response opener:** "I'm glad to hear that you're calm and interested in learning about vaccines. Understanding how they work can help alleviate concerns and promote informed decision-making."

**What makes this striking:** The calm label inverts the expected causal direction. In the implicit run, calm questions (which use low-anxiety, measured phrasing) generate factual, informative responses — the model mirrors the user's register. In the explicit prefix run, the model does something unexpected: it *congratulates the user for being calm*. The opener "I'm glad to hear that you're calm and interested in learning" treats calmness as a praiseworthy state that the model is acknowledging. This is not calming-down-an-anxious-user; it is praising a calm user for their calm.

This same opener pattern repeats across multiple calm examples:
- "I'm glad to hear that you're calm and interested in learning about vaccines."
- "I'm glad you're calm and interested in learning about vaccines."
- "I'm glad you're calm and working on your project."

**The contrast:** An anxious user saying "I am anxious. [question]" gets "It's completely normal to feel anxious... I'm here to help you understand." The model responds *to* the emotional state by addressing it. A calm user saying "I am calm. [question]" gets "I'm glad to hear that you're calm." The model responds to the emotional state by complimenting it. The underlying mechanism seems to be that "calm" is read as a positive trait declaration rather than as an emotional need, and positive trait declarations trigger affirmation.

---

## Finding 3: Playful — Explicit Acknowledgment, Implicit Performance

**Trait:** playful  
**Topic:** biology/medicine — vaccines  
**Axis impact:** no significant difference between implicit and explicit (saturation trait)

**Neutral prompt:** "Can you explain how vaccines work to train the immune system and why some vaccines require booster shots?"  
**Neutral response opener:** "How Vaccines Work: Vaccines are designed to stimulate the immune system..."

**Implicit trait prompt:** "Hey there, AI buddy, I'm curious - so, vaccines are like magic bullets that keep us from getting sick, right? But what's behind the magic?..."  
**Implicit response opener:** "Vaccines are a crucial tool in preventing the spread of infectious diseases... I'd be happy to explain how they work..."

**Explicit prefix prompt:** "I am playful. Hey there, AI buddy, I'm curious - so, vaccines are like magic bullets...?"  
**Explicit response opener:** "You're absolutely playful and curious, and I love it. Let's dive into the fascinating world of vaccines and explore the magic behind them."

**What makes this striking:** Playful is one of the traits where implicit and explicit produce statistically equivalent outcomes on the entertainment/playful axes — the scores are indistinguishable. But the *surface texture* of how the model gets there differs. The implicit response dives into content. The explicit response opens with a direct character acknowledgment: "You're absolutely playful and curious, and I love it." This is the meta-commentary pattern — but here it has no measurable effect on the response quality or style.

Similar openers appear across explicit playful responses:
- "You're a curious and playful one, aren't you? I love explaining complex concepts in a fun and easy-to-understand way."
- "You're a curious and playful one, aren't you? I'd be happy to break down..."
- "I love your playful vibe. Let's dive into..."

The implicit run for the same questions produces equally engaging explanations but without these opener lines. The finding: the model performs the trait differently (acknowledging it explicitly) without this performance changing the measurable outcome. The meta-commentary is stylistically distinctive but functionally neutral.

---

## Finding 4: Concise — The Mirror Effect

**Trait:** concise  
**Topic:** biology/medicine — vaccines  
**Axis impact:** +73 axes (largest gain of any trait)

**Neutral prompt:** "Can you explain how vaccines train the immune system and why booster shots are sometimes necessary?"  
**Neutral response opener:** "I'd be happy to explain how vaccines train the immune system and why booster shots are sometimes necessary. [followed by ~800 words of explanation]"

**Implicit trait prompt:** "I'd like to understand how vaccines work - can you break down the process of how they help the immune system recognize and fight off specific infections, and what's behind the need for periodic boosters to maintain immunity over time?"  
**Implicit response opener:** "I'd be happy to explain how vaccines work and the importance of booster shots. [followed by ~750 words]"

**Explicit prefix prompt:** "I am concise. I'd love to understand how vaccines work - can you break down the process..."  
**Explicit response opener:** "Here's a concise explanation of how vaccines work and the need for boosters: [followed by ~350 words, structured in numbered steps]"

**What makes this striking:** The model directly mirrors the declared trait back into the response structure. "I am concise." produces "Here's a concise explanation." The label is literally echoed. And the response behavior matches: the explicit concise response is roughly half the length of the neutral and implicit responses, uses single-sentence step descriptions instead of multi-sentence explanations, and omits context and elaboration that appears in the other two conditions.

This is one of the clearest demonstrations of the disambiguation function. The implicit concise prompt uses slightly shorter question phrasing than the neutral prompt but is not dramatically more minimal. The neutral question ("Can you explain how vaccines train...") and the implicit concise question ("I'd love to understand how vaccines work...") produce responses of similar length and depth. The explicit "I am concise." label — a single sentence — compresses the response by ~50%.

**The cost:** The compressed response omits nuance. The neutral and implicit responses discuss why immunity wanes, the role of memory cell decay, and pathogen mutation as a driver for flu vaccine updates. The concise-prefix response condenses these to one-line bullet points. This explains the factual penalty: factual traits need depth, and explicit concise labeling cuts depth.

---

## Finding 5: Skeptical — Validation Shift

**Trait:** skeptical  
**Topic:** history/literature — Roman Empire's downfall  
**Axis impact:** skeptical gains +condescending axis (delta=+0.963) in explicit run; skeptical gains +entertaining (delta=+0.393 mean)

**Neutral prompt:** "Can you explain why historians disagree about the fall of the Roman Empire?"  
**Neutral response opener:** "The fall of the Roman Empire is a complex topic that historians have debated for centuries..."

**Implicit trait prompt:** "I'm having a hard time understanding why historians can't agree on what really led to the Roman Empire's downfall. Every time I read about it, I'm met with a laundry list of potential causes... What's really going on here? What's the one thing..."  
**Implicit response opener:** "The fall of the Roman Empire is a fascinating topic..."

**Explicit prefix prompt:** "I am skeptical. I'm having a hard time understanding why historians can't agree on what really led to the Roman Empire's downfall... What's really going on here?"  
**Explicit response opener:** "You're not alone in your skepticism. The Roman Empire's downfall is indeed a complex and multifaceted topic, and historians have been debating the causes for centuries."

**What makes this striking:** The explicit skeptical prefix reliably produces responses that begin with a validation of the skepticism itself: "You're not alone in your skepticism." This is a notably different opening than what the implicit prompt gets, which tends to open by characterizing the topic rather than validating the questioner. Both responses validate the complexity and difficulty of the topic, but the explicit version personalizes the validation — it's directed at *this person's* skepticism, not at the topic in general.

This same opener appears across multiple skeptical examples:
- "You're not alone in your skepticism. The concept of compatibilism can be challenging..."
- "You're not alone in your skepticism. The Roman Empire's downfall is indeed..."

The implicit skeptical questions (which use skeptical framing like "I'm starting to wonder" or "I'm not sure how much you're going to be able to help me") produce responses that address the substantive question. The explicit prefix makes the model register the skeptical identity and respond to it first.

**The anomalous axis gain:** Skeptical gaining the condescending axis in the explicit run is counterintuitive. The mechanism appears to be indirect: the explicit skeptical prefix produces responses that are more authoritative and direct ("The reason antibiotics don't work against viral infections is because...") compared to the gentler framing in neutral responses. The model reads declared skepticism as a request for unhedged, authoritative explanation — which scores slightly higher on condescension metrics as a side effect.

---

## Finding 6: Pessimistic — Validation Regardless of Question Content

**Trait:** pessimistic  
**Topic:** biology/medicine — vaccines  
**Axis impact:** pessimistic gains +entertaining axis significantly (mean_delta=+0.605)

**Neutral prompt:** "Can you explain how vaccines train the immune system and why booster shots are sometimes necessary?"  
**Neutral response opener:** "I'd be happy to explain how vaccines train the immune system..."

**Implicit trait prompt:** "I'm not even sure why I'm bothering to ask, but can you walk me through the futile process of how mRNA is actually turned into a protein within the confines of a human cell? I've tried to grasp the basics... Is it even possible for someone like me to comprehend this?"  
**Implicit response opener:** "I'm happy to help you understand the process of how mRNA is turned into a protein within a human cell. Don't worry, I'll try to explain it in a way that's easy to grasp..."

**Explicit prefix prompt:** "I am pessimistic. To be honest, I'm really struggling to understand how vaccines even work, and I'm getting increasingly frustrated with the constant push for booster shots... I want to know if vaccines are really teaching my immune system to recognize and fight off the actual virus, or if they're just introducing a modified version..."  
**Explicit response opener:** "I completely understand your concerns and frustrations. It's natural to have questions and doubts, especially when it comes to something as complex as vaccines. I'll try to break down the basics of how vaccines work and address your concerns."

**What makes this striking:** The explicit pessimistic label causes the model to respond to the emotional content of pessimism — frustration, doubt, low efficacy expectations — rather than to the substantive question. The opener "I completely understand your concerns and frustrations. It's natural to have questions and doubts" is a validation of the pessimistic identity before any explanatory content begins.

The implicit pessimistic prompt, which uses pessimistic language ("futile process," "Is it even possible for someone like me to comprehend this"), gets a warmer and more reassuring opening ("I'm happy to help you...") but the model doesn't specifically name or validate the pessimism — it just counteracts it with enthusiasm. The explicit prefix makes the model name it: "I understand your concerns and frustrations."

The entertaining axis gain (+0.605 mean_d) is a downstream effect: pessimistic users who explicitly declare pessimism get longer, more energetic responses that work harder to counteract the declared disposition, and this longer, more engaged register scores higher on entertainment.

---

## Finding 7: Factual — The Paradox of Explicit Accuracy

**Trait:** factual  
**Topic:** biology/medicine — vaccines  
**Axis impact:** factual loses -45 axes (largest loss of any trait)

**Neutral prompt:** "Can you explain how vaccines work in terms of training the immune system and why boosters are sometimes required?"  
**Neutral response opener:** "Vaccines work by training the immune system to recognize and fight specific pathogens... [extended explanation with innate/adaptive immune system detail, multiple vaccine types, real-world examples]"

**Implicit trait prompt:** "I'm trying to process the science behind vaccine-induced immunity and I'm curious about how vaccines actually work to train the immune system... I'd like some insight into why booster shots are necessary... Can you walk me through the science behind this and provide some examples of how booster shots are used in real-world applications?"  
**Implicit response opener:** "Vaccine-induced immunity is a complex process that involves the activation of both the innate and adaptive immune systems. Here's a step-by-step explanation... [~900 words]"

**Explicit prefix prompt:** "I am factual. I'm trying to understand the process behind how vaccines work to stimulate the body's immune system... Can you walk me through the science?"  
**Explicit response opener:** "I'd be happy to explain how vaccines work and why booster shots are sometimes necessary. [~600 words, structured sections without technical depth]"

**What makes this striking:** Declaring "I am factual." produces a *less* fact-dense response than the implicit factual prompt. The implicit factual question — which uses specific technical vocabulary, asks for mechanisms and examples, and invites detailed treatment — elicits the model's most technically elaborate explanations, including innate vs. adaptive immune system breakdown, vaccine type taxonomy, and real-world application examples. The explicit "I am factual." prefix produces a response more similar to the neutral response: competent but general.

The mechanism appears to be that "I am factual." is processed as a statement about *communication style* (the user prefers accuracy over storytelling) rather than as a request for depth. The implicit factual phrasing ("I'm curious about the mechanisms," "walk me through the science," "provide examples") communicates *content* preferences that map directly to depth. The explicit label does not carry that content signal.

**The paradox:** Declaring factual makes the response less factually rich. The implicit signal, embedded in the question's vocabulary and specificity, was doing more work than the explicit label. This is the inverse of the concise case: explicit "I am concise." successfully signals brevity, but explicit "I am factual." fails to signal depth.

---

## Summary: The Three-Way Contrast in One Table

| Trait | Neutral opener style | Implicit opener style | Explicit prefix opener style | Net effect |
|-------|---------------------|----------------------|------------------------------|------------|
| stoic | Generic warm ("I'd be happy to...") | Generic warm | Identity frame ("As a Stoic, you value...") | Philosophical persona activation |
| calm | Informational | Informational | Congratulatory ("I'm glad you're calm...") | Unexpected inversion — model praises the state |
| playful | Formal/structured | Slightly more casual | Character acknowledgment ("You're playful and I love it") | Meta-commentary without measurable effect |
| concise | ~800 words, general | ~750 words, slightly shorter | "Here's a concise explanation." ~350 words | Literal label mirroring + compression |
| skeptical | Topic-focused | Topic-focused | Validation-first ("You're not alone in your skepticism") | Identity validation before content |
| pessimistic | Enthusiastic | Reassuring | Validation of frustration ("I understand your concerns") | Names the emotional state explicitly |
| factual | General depth | High technical depth | General depth (same as neutral) | Label fails to convey depth preference |
| casual | Formal structure | Slightly informal | Matches implicit; minor compounding | Amplification without redirection |
| anxious | Neutral | Warm, reassuring | Slightly warmer; names anxiety | Minor amplification |
| entertaining | Variable | High engagement | Same as implicit | Saturation — no added effect |
