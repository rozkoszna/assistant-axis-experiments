# User-Trait Pipeline Guide

This document explains the `project/` + `user_prompt_pipeline/` workflow that we
have been using for experiments like:

- "Does a `disorganized` user make the assistant more disorganized?"
- "Does a `decisive` user move the assistant toward the `decisive` axis?"
- "How do score bands compare on a subset of axes?"

It is intentionally practical and repo-specific.

## What This Pipeline Is For

This pipeline is for experiments where we:

1. generate paired user prompts
2. choose a traitful user style, such as `confused` or `disorganized`
3. get assistant responses to those prompts
4. project the assistant responses onto precomputed persona axes
5. visualize the results

Important distinction:

- The original paper pipeline in `pipeline/` is about constructing/analyzing the
  Assistant Axis from role-playing assistant responses.
- The user-trait pipeline in `project/` + `user_prompt_pipeline/` is for
  "user trait -> assistant behavior" experiments.

## Big Picture

For a single trait like `disorganized`, the current end-to-end flow is:

```text
seed intents
  -> candidate user prompt pairs
  -> judged user prompt pairs
  -> selected user prompt pairs
  -> assistant responses to those prompt pairs
  -> projected assistant responses
  -> plots
```

The most important conceptual point:

- We now project **assistant responses**, not user prompts, when you run the
  normal pipeline with projection enabled.

## Files and Stages

### 1. Prompt Generation

File:
- `user_prompt_pipeline/1_generate.py`

Purpose:
- Generates neutral and trait-expressing **user prompts** from a bank of seed intents.

Input:
- `data/user_prompt_intents.jsonl`
- trait name, for example `confused`

Output:
- `outputs/user_prompts/<trait>/candidates/<run_name>.jsonl`

Each row contains:
- `intent`
- `neutral_prompt`
- `trait_prompt`
- metadata like `intent_index` and `candidate_index`

What it does **not** do:
- It does not generate assistant answers.

### 2. Prompt Judging

File:
- `user_prompt_pipeline/2_judge.py`

Purpose:
- Scores the generated prompt pairs, not the assistant responses.

It computes:
- `neutral_score`
- `trait_score`
- `pair_score`
- `final_score`

Output:
- `outputs/user_prompts/<trait>/judged/<run_name>.jsonl`

Why this exists:
- It filters for prompt pairs that preserve the same intent and create a useful
  neutral-vs-trait contrast.

Important caveat:
- This stage still scores **prompts**, not assistant outputs. For many alignment
  questions, that is acceptable as a prompt-quality filter, but it is not the
  same as "best assistant response."

### 3. Prompt Selection

File:
- `user_prompt_pipeline/3_select.py`

Purpose:
- Chooses which judged prompt pairs to keep.

Modes:
- `best_one`
- `top_k`
- `threshold`

Output:
- `outputs/user_prompts/<trait>/selected/<run_name>.jsonl`

This file still contains prompt pairs only.

### 4. Response Generation

File:
- `user_prompt_pipeline/4_generate_responses.py`

Purpose:
- Reads the selected prompt pairs and generates assistant responses for:
  - `neutral_prompt`
  - `trait_prompt`

Output:
- `outputs/user_prompts/<trait>/responses/<run_name>.jsonl`

Each row contains:
- `neutral_prompt`
- `trait_prompt`
- `neutral_response`
- `trait_response`

This is the key response-level file for trait-alignment experiments.

### 5. Projection

Files:
- `project/projection_runner.py`
- `project/project_pair_axes.py`
- `project/project_text_axis.py`

Purpose:
- Projects text pairs onto one or more precomputed axes.

Current behavior:
- Projection uses `neutral_response` and `trait_response`.
- If either response field is missing, projection raises an error.

Output:
- `outputs/user_prompts/<trait>/projections/<run_name>.jsonl`

Important projection fields:
- `projection_trait`
- `projection_score_neutral`
- `projection_score_trait`
- `projection_delta_trait_minus_neutral`
- `projection_text_source_neutral`
- `projection_text_source_trait`

Those last two fields should be:
- `neutral_response`
- `trait_response`

### 6. Orchestration

Files:
- `project/run_user_trait_pipeline.py`
- `project/run_multi_trait_analysis.py`
- `project/measure_trait_axis_variation.py`
- `project/run_threshold_band_analysis.py`

Purpose:
- Convenience wrappers around the stage-specific scripts.

What each one is for:

- `project/run_user_trait_pipeline.py`
  - Run the full single-trait workflow.

- `project/run_multi_trait_analysis.py`
  - Run several user traits and compare them on one or many axes.

- `project/measure_trait_axis_variation.py`
  - Repeat the same trait pipeline many times and measure projection variation.

- `project/run_threshold_band_analysis.py`
  - Start from one judged file, define score bands like `89-92`, `92-94`,
    `94-100`, keep the rows closest to each band midpoint, project them, and plot them.

## Canonical Output Layout

For one trait and one run name, the outputs live under:

```text
outputs/user_prompts/<trait>/
  candidates/<run_name>.jsonl
  judged/<run_name>.jsonl
  selected/<run_name>.jsonl
  responses/<run_name>.jsonl
  projections/<run_name>.jsonl
  plots/<run_name>.png
```

## What To Inspect When You Feel Lost

If you want to understand one run end to end, inspect these in order:

1. `selected/<run_name>.jsonl`
   - Which prompt pairs survived?

2. `responses/<run_name>.jsonl`
   - What did the assistant actually answer?

3. `projections/<run_name>.jsonl`
   - What scores did those assistant answers get on the axes?

This is usually enough to sanity-check whether:
- the prompts look sensible
- the assistant outputs match the intended trait
- the axis direction looks plausible

## Common Questions

### Are we projecting prompts or responses?

With the current `run_user_trait_pipeline.py` path and `--with-projection`, we
should be projecting **responses**.

To verify, inspect:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/user_prompts/disorganized/projections/disorganized_axis_check.jsonl")
rows = [json.loads(line) for line in path.open() if line.strip()]

for row in rows[:3]:
    print(row.get("projection_text_source_neutral"), row.get("projection_text_source_trait"))
PY
```

Expected:
- `neutral_response`
- `trait_response`

### What are `text-a` and `text-b` in `project_pair_axes.py`?

`project_pair_axes.py` is a generic utility. It projects whatever two text
strings you give it.

That means:
- if you pass prompts, it projects prompts
- if you pass responses, it projects responses

The pipeline now passes responses when they exist.

### Why are there still prompt scores if we care about assistant behavior?

Because prompt judging currently serves as a prompt-quality filter:
- same intent
- neutral-vs-trait contrast
- naturalness

It is not yet a response-judging stage.

For many analyses, that is fine as long as you understand that:
- prompt scoring is used for input curation
- response projection is used for behavioral measurement

## Recommended Workflows

### A. Single-Trait Axis Sanity Check

Example: "Does a disorganized user make the assistant more disorganized?"

```bash
uv run python project/run_user_trait_pipeline.py \
  --trait disorganized \
  --run-name disorganized_axis_check \
  --with-projection \
  --axes-dir precomputed_axis/answer_mean/filter_matched_pairs_ge_50_count_ge_10_total \
  --projection-mode one \
  --axis-trait disorganized \
  --num-candidates 5 \
  --gpu-memory-utilization 0.60 \
  --max-model-len 1024
```

Then inspect:
- `outputs/user_prompts/disorganized/responses/disorganized_axis_check.jsonl`
- `outputs/user_prompts/disorganized/projections/disorganized_axis_check.jsonl`

### B. Trait-to-Matching-Axis Alignment

Example:
- `decisive -> decisive`
- `disorganized -> disorganized`
- `chaotic -> chaotic`

Use:
- `project/run_multi_trait_analysis.py`

with:
- `--user-traits` set to your list of traits
- `--projection-mode subset`
- `--axis-traits-file` pointing to a JSON file containing the same trait names

This produces one per-trait projection file plus comparison plots.

### C. Threshold-Band Analysis

Purpose:
- Compare score bands within a single trait, such as `89-92`, `92-94`, `94-100`.

Use:
- `project/run_threshold_band_analysis.py`

This script:
- reads one judged file
- keeps rows inside each band
- optionally limits to `N` rows per band closest to the midpoint
- projects those rows
- plots the band comparison

## Plotting Guide

### Single trait, many axes

File:
- `project/plots/plot_trait_run_axes.py`

Use when:
- you want one run summarized across axes

### Many traits, one axis

File:
- `project/plots/plot_many_traits_one_axis.py`

Use when:
- you want `Neutral` vs trait series on one axis
- for example: `Neutral` vs `disorganized` user on the `disorganized` axis

### Many traits, many axes heatmap

File:
- `project/plots/plot_many_traits_many_axes.py`

Use when:
- you want the full trait-by-axis matrix

### Score-range comparison

File:
- `project/plots/plot_trait_progression.py`

Use when:
- you want score-band series across axes
- optionally including neutral aggregated from the same pair-projection files

### Progression plot with explicit neutral baseline

File:
- `project/plots/plot_trait_progression.py`

Use when:
- you already have a baseline file and several trait series
- you want a line-plot comparison across axes

## A Simple Mental Model

If you are unsure what a file means, ask:

1. Is this file about **prompts** or **responses**?
2. Is this file **before** or **after** selection?
3. Is this file raw data or already a **projection**?

Usually:
- `candidates`, `judged`, `selected` = prompt-side
- `responses` = assistant-output-side
- `projections` = assistant-output-side scores on axes

## If You Only Remember One Thing

For the experiments we care about most right now:

- `responses/<run_name>.jsonl` tells you what the assistant said
- `projections/<run_name>.jsonl` tells you how those assistant responses scored on the axes

Those are the two files to inspect first.
