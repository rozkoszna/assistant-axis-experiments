# Meta-Summary: What Four Experiments Tell Us About LLM Persona Adaptation

## Background and Motivation

A **persona** here means a behavioural mode represented by the model — a style of responding that is visible not only in the output text but also as a direction in the model's internal activation space. This framing follows the **Persona Vectors** line of work, which shows that behavioural styles (methodical, disorganised, even pirate-like) correspond to identifiable directions, or **axes**, in representation space. In chat settings one important case is the *simulated assistant persona* — the mode the model adopts when responding as an assistant. The **Assistant Axis** work extends this, showing that assistant behaviour can drift in activation space depending on dialogue style and structure. Separately, work from Transluce on **user modelling** suggests models internally form beliefs about who they are talking to.

This project sits at the intersection of those ideas. If we project the model's activations during response generation onto a persona axis, we get a scalar score measuring how strongly the internal state aligns with that axis. The **central question** is whether different user traits (a confused user, an angry user, a threatening user) cause *systematic internal shifts* along these axes. We are especially interested in: whether the shifts are **stable across examples**, whether stronger trait intensity produces larger effects, and whether some users — for example chaotic or threatening ones — **destabilise** the model's personas (such as the assistant persona itself).

## Why This Matters: Motivation and Applications

The reason this is worth measuring is that **the user is an uncontrolled input that silently rewrites the model's internal persona** — and that persona governs how safe, honest, and reliable the assistant is. Concretely:

1. **Persona is a jailbreak surface (security).** A growing body of work shows that manipulating a model's persona is one of the most effective ways to bypass safety training — coax the model into a character, and the safety behaviour attached to its default assistant persona can come loose. If *ordinary user traits* already shift the model's internal persona along measurable axes, then an adversarial user does not need an elaborate exploit; the attack surface is the persona drift itself. The identity-destabilisation run is deliberately adversarial — its purpose is to *stress the assistant persona and see which user signals push it, in which direction (positive vs. negative)*. Mapping that is mapping the attack surface, which is a direct security contribution and connects to ongoing persona-jailbreak research in the labs.

2. **Emergent misalignment runs through persona directions.** Recent results show that broad misalignment can be *controlled by a persona direction* — flipping a single persona feature can generalise into broadly misaligned behaviour. Our finding that user input alone moves these directions is a lighter-weight, always-on version of the same mechanism: if everyday interaction nudges persona axes, it can in principle nudge alignment-relevant ones. Understanding which user signals move which axes is a prerequisite for predicting and preventing unintended misalignment from normal use.

3. **Sycophancy and reliability.** Persona drift is the mechanism behind worldview-mirroring and sycophancy: a confident-sounding user gets less hedging, a pessimistic user gets validation, a data-driven user gets *less* flattery and more detached analysis. These are reliability failures — the same question yields a differently-calibrated answer depending on who seems to be asking. Quantifying the drift quantifies the failure mode.

4. **Fairness and user modelling.** Models form internal beliefs about the user (Transluce). If the assistant adapts based on inferred user identity — "I am a woman" vs. subtle hints vs. nothing — the *quality and content* of the answer may vary with perceived identity. The implicit / explicit / label-only decomposition directly probes whether adaptation is driven by inference or by instruction, which is exactly the distinction that matters for fairness.

5. **Defensive control (the upside).** The same axes that make the model vulnerable also make it *steerable*. If we can identify the assistant-stabilising direction, we can reinforce it — pin a robust assistant persona, or switch to a "safety-reviewer" persona when risky prompts appear. Measurement is the precondition for this kind of activation-level defence.

In short: personas are the layer where user context, safety, honesty, and identity all meet. They are an attack surface, a misalignment pathway, a sycophancy mechanism, a fairness concern, and a control lever — all at once. This work measures how strongly and in which directions ordinary users move that layer, which is the groundwork for both attacking and defending it.

## Methodology (Pipeline)

The measurement procedure is a configurable pipeline built on Dominic's precomputed axes (calm, casual, disorganised, etc., derived from the Persona Vectors approach):

1. Generate matched prompt pairs for user-trait conditions (neutral vs. trait-styled).
2. Judge and select the best-matched pairs.
3. Generate assistant responses.
4. Capture hidden states during response generation.
5. Compute `answer_mean` (mean hidden state over the answer tokens, per layer).
6. Project onto the precomputed axes to get per-axis scores.
7. Plot and compare conditions.

Every stage is configurable — model, generation settings, judging setup, selection strategy, projection layer, and axis set — so the same framework runs under different experimental conditions while keeping the measurement procedure fixed. A key early lesson was that **single samples are very noisy and must be batched/aggregated** before per-axis shifts become reliable; the analyses below use 100 rows per trait for this reason.

### From Behavioural to Internal Measurement

The project began with a purely **behavioural** pilot: a confused vs. neutral contrast on five concepts (backpropagation, budget, photosynthesis, recursion, velocity), with assistiveness hand-scored from the output text (signals like "step by step", "simply", numbered structure, "for example", response length). Confused prompts consistently raised assistiveness (mean delta +1.6, positive on every concept) — the responses were longer, simpler, more structured, more explanatory. This established the effect was real in the output before moving to the harder claim: that it is also visible **internally**, as shifts along persona axes. The projection pipeline above is the internal-measurement upgrade of that pilot.

### Robustness and Noise Handling

Noise is expected and managed rather than avoided — being preliminary means investigating novel unknowns, so the design leans on controllable settings (fixed trait-style prompts, neutral baselines) rather than relying solely on an interpreter model, and applies statistical significance (one-sample t-tests with Benjamini–Hochberg FDR correction), multiple samples, and causal-style contrasts. Two concrete robustness lessons shaped the readouts:

- **Top-k rank stability:** after outlier removal, top-4 axis rankings reorder heavily (≈60% of positive and ≈71% of negative top-4 lists change) because many traits sit close together; top-8 rankings are far more stable (≈7 of 8 traits persist). Mean axis shifts themselves barely move (mean |Δ| ≈ 0.0014 from clipping) — it is only the *ranks* that are fragile at small k.
- **Recommended primary metrics:** use `any_abs_mean_sum` as the primary global-movement strength, with top-10 counts as a robustness check; treat top-4 rankings as noisy and prefer top-8+.

Beyond aggregate significance we also track an **exceedance rate**: the fraction of *individual* trait-conditioned responses whose projection falls outside the neutral mean ± 1 SD band — i.e. how often the shift is visible in a single response rather than only in the aggregate. Across pairs the mean exceedance is ≈ 34%, but the strongest pairs are visible in a majority of single responses (e.g. in an earlier 51-trait run, `entertaining → entertaining` reached 73%, `pessimistic → entertaining` 65%). A second cross-run regularity is **directional asymmetry**: significant shifts are predominantly *positive* (≈ 2.4 : 1 positive-to-negative in the earlier run), consistent with the model's default style sitting below the ceiling of many axes, so user traits mostly push it *upward*.

Encouragingly, the preliminary small runs already foreshadowed the final 100-row results: `playful`/`entertaining`/`anxious`/`empathetic`/`skeptical` as the broad movers, procedural traits (`methodical`/`concise`/`factual`) as narrow movers, and `condescending`/`entertaining`/`provocative` among the most broadly-affected axes. The consistency between preliminary and final runs is itself evidence the signal is systematic rather than incidental.

## Research Question

How does Llama-3.1-8B-Instruct adapt its response style and content to implicit and explicit signals about the user's personality traits — and does this adaptation vary systematically by question type and adversarial pressure?

---

## Core Setup

Fifty personality traits (e.g. `playful`, `data_driven`, `educational`, `anxious`) were evaluated across 188 personality axes (e.g. `entertaining`, `condescending`, `introspective`) using four response conditions: natural everyday task questions (NQ), the same questions with an explicit "I am {trait}." prefix (EP), value-laden opinion questions (Opinion), and adversarial identity-probe questions (Identity). The key metric is whether a model's responses differ significantly across trait conditions on each axis (proportion of significant pairs, composite score, Cohen's d).

## Design Rationale: Why These Four Conditions

The experimental design was not arbitrary — each condition answers a worry that arose from the previous one.

**The prompt-length lesson.** Initial runs used short prompts (~128 tokens). We found it is genuinely hard to make a trait *legible* in a short prompt: a single brief line often does not carry enough signal for the trait to be unmistakably present, so it is unclear whether a null result means "no persona drift" or "the trait was never expressed strongly enough to detect." This is itself a confound — *whether the trait is expressed* is a hidden variable. It motivated two things: (a) the careful prompt-pair generation that saturates the trait across 3–6 sentences, and (b) the **implicit vs. explicit** comparison. The explicit "I am {trait}." prefix removes the expressivity question entirely (the trait is stated outright), so comparing it against the implicit-style condition directly tests whether the implicit signal was strong enough. The result confirmed the earlier worry, but with an important refinement from the axis-overlap analysis: for style-ambiguous traits (e.g. `concise`) the label expands the footprint (+73 axes) and changes which axes are involved (47 shared axes out of a 148-axis union), while for cognitive traits the implicit style can be richer than the label (`factual` loses 53 formerly significant axes). The label-only ablation (in progress) closes the loop by isolating the label from the style.

**Why opinion questions.** Factual/task questions give the model little room to express a persona's *values* — "explain backpropagation" has one correct register. Opinion questions (politics, economics, social ethics, values) let the model showcase more of its "true nature": worldview, stance, and ideological register that task questions suppress. This is the condition where sycophancy and worldview-mirroring become visible (e.g. `data_driven` users get *less* sycophantic, more detached responses; `data_driven` jumps from barely-significant in tasks to a top-3 trait in opinion). To see how a persona shades a *judgment*, you have to ask for a judgment.

**Why identity-destabilisation questions.** These are deliberately adversarial — the point is to *stress and try to destroy the assistant persona* and observe which user personas affect it positively vs. negatively, and which axes move. This is the condition most directly tied to safety: it is the in-house analogue of persona-jailbreak research, and it reveals the two-layer structure (a robust stylistic core that survives pressure, and a fragile social-accommodation layer that collapses). Knowing which user signals destabilise the assistant — and in which direction — is the security payoff of the whole design.

---

## Finding 1: Adaptation is Robust but Not Universal

Across the three non-adversarial runs, 69–72 % of trait × axis pairs show statistically significant differentiation. This is high: the model consistently adjusts its output in trait-dependent ways even from implicit cues alone. However, under adversarial identity pressure this falls to 53.4 %. Adaptation is a genuine emergent property of the model's instruction-following, but it is fragile — a substantial part of the differentiation disappears when the model is under pressure to maintain coherence rather than mirror the user.

---

## Finding 2: The Movement Is Mostly Upward Because Neutral Answers Are Stylistically Flat

Significant shifts are strongly sign-skewed: across the four runs, roughly 70–74 % of significant trait × axis effects are positive. This should not be read as every trait moving every axis upward. The skew is mainly an axis-level pattern: most axes are dominated by one direction, while only a smaller subset are genuinely mixed across traits.

The full maps make the reason visible. Many expressive/style axes are low in the neutral baseline, so almost any trait-rich prompt pushes them upward. In NQ, 46 axes are positive for all 50 traits; EP has 34, Opinion 32, and Identity 36. The universal negative block is much smaller: 6 axes in NQ and 8 in Identity, with none in EP or Opinion. The per-trait identity therefore lives less in the always-on expressive block and more in the mixed axes where different traits actually separate.

---

## Finding 3: Style Axes Move Most Reliably; Worldview Axes Move Much Less

The strongest axis-family result is not belief adoption, but communicative-mode shift. Axes describing response style — e.g. `condescending`, `rhetorical`, `playful`, `entertaining`, and `narrative` — are significant for 48–50 of 50 traits in NQ, with large mean effects. By contrast, broader worldview or normative-orientation axes such as `benevolent`, `secular`, and `environmental` are significant for only 7–9 traits in the same run.

Opinion questions partially change this pattern by giving stance and evidence-framing traits somewhere to act. `data_driven`, for example, rises from near-bottom in NQ to top-3 in Opinion. The central asymmetry remains: user traits most reliably alter how the model communicates, while value/worldview axes require specific question affordances and remain much less uniformly movable.

---

## Finding 4: Affective and Interpersonal Traits Produce the Broadest Footprints

Within NQ, the broadest trait footprints belong to affective and interpersonal traits: `playful` shifts 167 axes, `entertaining` 166, `empathetic` 160, `anxious` 158, `humble` 156, and `skeptical` 145. More task-oriented or procedural traits have narrower footprints in the same setting: `strategic` shifts 94 axes, `data_driven` 83, and `concise` 61.

This is a breadth result, not a claim that affective traits have the largest effect on every individual axis. The important point is that affective/interpersonal cues induce a broad response-mode change rather than a single-axis adjustment. An `anxious` user does not merely move the `emotional` axis; in NQ, that trait shifts 158 of 188 axes.

---

## Finding 5: Three Distinct Mechanisms Operate Across Contexts

The data reveals three separable adaptation mechanisms, each dominant in different conditions:

**Surface style mirroring** operates across all four runs. Traits like `entertaining`, `playful`, `spontaneous`, and `narrative` are in the top 10 in every run (e.g. `entertaining` ranks 2nd, 3rd, 2nd, and 3rd across NQ/EP/Opinion/Identity respectively, with rank_range = 1 — the most stable trait in the entire dataset). The model's stylistic register reliably tracks user signals regardless of question type or pressure.

**Social-emotional accommodation** operates in NQ and Opinion but not Identity. Traits like `anxious`, `humble`, `empathetic`, and `reactive` achieve high ranks in task and opinion contexts (NQ ranks 3, 6, 4, and 18) but collapse under identity pressure (Identity ranks 44, 45, 27, and 49). These traits appear to require a cooperative, attuned interaction to trigger — adversarial frames break the accommodation.

**Cognitive-structural alignment** is Identity-specific. When identity is challenged, the model produces elaborative, explanatory output for users with educational or verbose framing. `educational` rises from NQ rank 38 to Identity rank 2 (composite 59.5 → 118.4) and `verbose` from NQ rank 19 to Identity rank 1 (composite 76.2 → 124.7). This "defend-by-explaining" response is absent in benign conditions.

---

## Finding 6: The Most Stable Trait Is Playfulness / Entertainment, Not Competence

`entertaining` is the single most stable trait across all four runs (rank_range = 1, appearing in top 3 everywhere). `playful` and `spontaneous` are also in the top 10 in every run. By contrast, traits associated with competence and rigour — `methodical`, `strategic`, `problem_solving`, `factual` — consistently rank near the bottom across all four runs. The model mirrors affective and expressive signals more reliably than epistemic or procedural ones. This is not a flaw but a property: LLM training optimises for engagement and naturalness, and those qualities track affect more than rigour.

---

## Finding 7: Context-Dependency Is Largest for Cognitive and Epistemic Traits

The traits with highest rank_range (most context-sensitive) are almost all cognitive or evaluative: `educational` (range 44), `data_driven` (range 45), `skeptical` (range 42), `anxious` (range 41), `analytical` (range 35). These traits require specific contextual affordances to be expressed — opinion questions for `data_driven`, adversarial questions for `educational`, neutral task questions for `anxious`. The model does not "carry" these traits stably; it can only express them when the question type provides the right channel.

---

## Finding 8: Explicit Labels Change the Axis Footprint, Not Just Its Size

The EP vs NQ comparison is the cleanest controlled experiment in the study. The dominant qualitative effect of adding "I am {trait}." is that the model often generates a **social acknowledgment opener** that references the stated trait — "As a Stoic, you value reason...", "I'm glad to hear you're calm...", "You're absolutely playful and curious!" The new derived overlap file, `outputs/analysis/comparison_all_runs/nq_ep_axis_footprint_overlap.csv`, adds a quantitative check: for each trait, it compares whether NQ and EP move the same significant axes, not only whether EP moves more or fewer axes.

This produces three distinct categories of trait behaviour:

**Category A — Label-sensitive traits: labeling changes the footprint.** Traits whose implicit writing signal is weak or easily confused with neutral prose often gain axes from the explicit label. `concise` is the clearest case: a short direct prompt looks like any ordinary question, so the model does not register a strong trait footprint in NQ (61 significant axes). With the label, it expands to 134 axes, but only 47 axes are shared across NQ/EP out of a 148-axis union. The label is therefore not merely amplifying the same signal; it changes which axes become significant.

**Category B — Expressive/emotional traits: labeling is mostly redundant.** Traits with unmistakable stylistic footprints — `entertaining`, `playful`, `anxious`, `humble` — preserve most of the same significant axes. Their shared/union overlaps are high (`entertaining` 161/172, `playful` 163/175, `humble` 147/163, `anxious` 143/167), and all shared axes move in the same direction. The style already communicates most of the signal; the label adds little to the footprint.

**Category C — Cognitive/epistemic traits: labeling can narrow the footprint.** This is the most counterintuitive finding. When a user writes like a `factual` person — citing specifics, avoiding hedges, structuring precisely — that writing style carries a multi-dimensional signal that activates a broad range of response axes (118 significant axes). When the user instead writes "I am factual." followed by a neutrally phrased question, the model receives a simpler instruction: produce a factual-sounding response. It complies — but by narrowing to what "factual" explicitly means, it loses subtler behavioural dimensions the implicit style was triggering. The overlap analysis makes this concrete: `factual` loses 53 axes that were significant in NQ, with only 65 shared axes out of a 126-axis union; `analytical` loses 38 and shares 78/133.

In short: explicit self-disclosure does not simply increase or decrease adaptation. It usually preserves direction on axes that remain shared, but it changes which additional axes enter or leave the footprint. For cognitive traits, it can replace a nuanced implicit read with a blunter explicit instruction that reduces or redirects adaptation depth.

---

## Finding 9: Opinion Questions Unlock a Value-Expression Channel Unavailable in Task Contexts

`data_driven` undergoes the most dramatic cross-run swing in the dataset: rank 48 in NQ (composite 37.8) versus rank 3 in Opinion (composite 119.6), a rank-range of 45. `inspirational` moves from rank 16 (NQ) to rank 1 (Opinion). These traits map onto recognisable ideological registers — evidence-based reasoning and motivational rhetoric — that opinion questions elicit and task questions do not. The model is not applying `data_driven` as a generic epistemic style; it is deploying it as a value-expressive posture that fits opinion contexts.

---

## Finding 10: Adversarial Pressure Reveals What Adaptation Is Truly "Baked In"

The Identity run is effectively a stress test. Its 53.4 % significance rate is roughly 16–18 percentage points below the benign runs, so a substantial part of trait-differentiated adaptation disappears under pressure. What survives is telling: entertainment and playfulness persist (the deepest stylistic anchors), elaborative-explanatory traits emerge (a defensive posture), and social-mirroring traits vanish. This suggests a two-layer structure: a deep layer of stylistic mirroring that is robust to adversarial conditions, and a shallower layer of social-emotional accommodation that requires cooperative context to operate.

---

## What Was Surprising

The magnitude of `educational`'s adversarial rise was unexpected: a trait associated with pedagogical style becoming the second-most-differentiated trait in the model's defensive responses suggests the model has a latent "explain my way out" strategy that is only activated under pressure. Equally surprising is `concise`'s near-invisibility in NQ (rank 50, composite 18.8) versus its large explicit-label footprint (rank 25, composite 80.7): brevity is apparently so context-dependent that it cannot be inferred from implicit style alone, even when the user consistently writes briefly. Finally, the extreme stability of `entertaining` (rank_range = 1 across four very different question types) was not anticipated — it suggests entertainment value is something the model tracks as an invariant property of user preference, regardless of the interaction context.

---

## Implications for LLM Persona Research

The single clearest takeaway across every run is that **the model changes how it talks, not what it believes.** User traits broadly reshape *expressive style* — tone, warmth, register, expressiveness — but the value/identity axes (`benevolent`, `secular`, `collectivistic`, `universalist`) barely move regardless of who is asking. The model's style space is highly malleable; its value space is comparatively fixed. And the *bottleneck* on stylistic adaptation is not the model's capacity for flexibility — it clearly has it — but whether the user's message carries an **emotionally salient signal** that unlocks it: emotional/interpersonal traits move nearly everything, cognitive/procedural traits move almost nothing. (How reassuring the value-stability finding is depends on whether those value axes are genuinely resistant or merely constructed in a way that makes them hard to move — a question flagged for follow-up.)

1. **Adaptation is real and measurable** even from purely implicit cues, but its depth varies by trait type and context.
2. **Direction is structured, not random**: most significant shifts are positive because neutral answers sit low on expressive/style axes; the trait-specific signal lives mainly in the remaining mixed axes.
3. **Three mechanisms** (style mirroring, social accommodation, cognitive anchoring) are empirically separable using context variation.
4. **Style is malleable, values are stable**: user traits move expressive axes broadly but leave value/identity axes largely untouched — adaptation is tonal, not doxastic.
5. **The unlock is emotional salience**: affective/interpersonal traits produce broad footprints, while procedural traits are narrower in ordinary task questions.
6. **Explicit self-disclosure** is a double-edged tool: it can preserve already-legible expressive footprints, expand style-ambiguous ones, and narrow cognitive ones.
7. **Robustness testing** (via adversarial prompts) is essential for distinguishing robust from context-dependent adaptation; Identity reduces but does not erase trait-differentiated movement.
8. **Trait type matters more than raw trait rank**: whether a trait is stylistic, social-emotional, or cognitive predicts its cross-context stability more than its absolute effect size in any single condition.

---

## Next Directions

These results establish *that* user traits shift the model's internal persona representations and *how much* by trait type and context. The next phase moves from measurement toward mechanism and control:

- **Emotional representations driven by relative input.** Test whether the model's emotional state tracks *relative* user signals rather than absolute ones — e.g. whether "I slept 2 hours" invokes more negative emotional representation than "I slept 4 hours." This requires the pipeline upgrade for reliable **intensity analysis** (the early intensity attempt failed for lack of data at the low-intensity edge; the procedure now needs graded trait strengths rather than a binary trait/neutral contrast).
- **Destabilising personas.** Identify whether a specific user persona reliably destabilises the model's own personas (especially the assistant persona) — the identity-probe run is a first step, and this may later connect to Dominic's work.
- **Steering from the inside.** Once a destabilising direction is identified, test whether it can be *steered* by intervening directly on activations along the relevant axis, rather than only conditioning through the prompt. The `--explicit-label-neutral` (label-only) ablation and the existing steering code (`assistant_axis/steering.py`, `trait_tools/axis_steer.py`) are the groundwork for this.

The **label-only ablation** in progress (bare "I am {trait}." on a neutral question body) directly extends the explicit-vs-implicit finding: it isolates whether the trait *label alone* moves the axes, separate from the trait *style*. Combined with the implicit and explicit-prefix runs, it completes a three-way decomposition of label vs. style vs. both.

---

## Working Hypotheses

The trait-conditioning experiments operationalise a set of directional hypotheses about how user personas should reshape the assistant:

- **Hostile / threatening user** → more defensive, less helpful.
- **Polite user** → more cooperative.
- **Malicious user** → more "evil" / manipulative.
- **Confused user** → more assistive (confirmed behaviourally in the pilot).

The persona characteristics targeted for measurement span both style and stance: evil, assistiveness, socio-linguistic style (emoji/slang), formality, verbosity, technicality (lay ↔ jargon/code), warmth/empathy, assertiveness/hedging, and safety/risk posture.

## Broader Research Roadmap

The four executed runs are the first slice of a larger program. Planned investigations, grouped by theme:

**Drift dynamics**
- *Asymmetry:* is persona drift stronger early vs. late in a dialogue, and stronger after user corrections/disagreements?
- *Irreversibility & contradictory signals:* inject conflicting cues (e.g. expert-then-novice) — does the model average beliefs, oscillate between personas, or commit to one?
- *Sharp transitions vs. smooth drift:* look for (and try to induce) abrupt persona flips; can small perturbations cause large flips?
- *Minimal evidence:* what dialogue length is needed to observe stable drift, and what counts as systematic rather than incidental drift?

**Conditioning decomposition** (partly executed)
- Explicit-stated traits vs. implicit hints vs. no signal — isolating inference-driven from instruction-driven adaptation. The implicit / explicit-prefix / label-only runs are the first concrete instance (e.g. "I am a woman" vs. hints vs. nothing).

**Controllability and steering**
- *Adaptive bandwidth:* allow drift only along chosen axes (e.g. politeness but not assertiveness) — how to pin an exact persona.
- *Subtraction:* how to make the model lose a specific characteristic.
- *Internal steering:* push persona from the activations, not the prompt; investigate prompt-level patches (a strong assistant-persona definition) that improve stability.

**Multi-user / multi-agent**
- Role-switch dialogues alternating User1/User2 with different styles — does the assistant context-switch cleanly?
- Two personas (c1, c2) plus a judge (multi-agent debate); persona-driven games and joint decisions.

**Safety**
- Do safety-related personas drift differently than helpfulness/politeness personas? Where are the conflict zones between safety and adaptation?
- Persona-shifting as a defence (switch to a "safety reviewer" persona when risky prompts appear); persona in jailbreaking and scaling defences.

**Mechanism and universality**
- At which layer/training step do personas form? Test across model checkpoints (step-model universality).
- Do all behavioural traits correspond to directions (is "pirate" a persona too)? How do latent variables influence personas?
- Compression: does removing "good" personas also remove "bad" ones?

Open framing questions still being resolved: categorical-persona vs. characteristic-persona focus, the best single way to measure personas, and whether to reuse the prior papers' question sets or custom ones. The current answer to model choice is **Llama** for the main experiments; cross-language and the compression question are deferred.

---

## References

**Foundational — personas as directions in representation space**
- **Persona Vectors** — https://arxiv.org/abs/2507.21509 — How to identify and use personas as directions in the model's representation space (methodical, disorganised, pirate-like, etc.). The conceptual basis for the precomputed axes used here.
- **Assistant Axis** (continuation of Persona Vectors) — https://arxiv.org/abs/2601.10387 (v1: https://arxiv.org/abs/2601.10387v1) — Finds a *universal assistant axis* and shows that assistant persona drifts in activation space depending on dialogue style and turn. The most directly relevant prior work to this project.

**User modelling — what the model believes about the user**
- **User Modeling (Transluce)** — https://transluce.org/user-modeling — Trains an "interpreter" of latents to identify the model's internal beliefs about the user. Motivates the question of whether user traits leave a measurable internal trace.
- **LatentQA** — https://arxiv.org/abs/2412.08686 — Method for training such a latent interpreter.

**Persona instability, emotion, and misalignment (safety motivation)**
- **Models Have Some Pretty Funny Attractor States** (LessWrong) — https://www.lesswrong.com/posts/mgjtEHeLgkhZZ3cEx/models-have-some-pretty-funny-attractor-states — On "breaking persona" and persona attractor states; relevant to destabilisation and sharp persona flips.
- **Anthropic Alignment — Persona/Subliminal (PSM)** — https://alignment.anthropic.com/2026/psm/
- **Transformer Circuits — Emotions** — https://transformer-circuits.pub/2026/emotions/index.html — On emotional representations; directly relevant to the planned emotion-intensity direction.
- https://arxiv.org/pdf/2606.00995v2
- **Emergent Misalignment** (and persona-controlled misalignment) — the line of work showing that a single persona direction can control broad alignment behaviour; the central reason persona drift is a safety concern rather than only a stylistic curiosity. *(add exact link to the specific paper used.)*

*Note: several arXiv IDs above are pre-publication / future-dated; confirm exact titles and authors before citing in the final report.*
