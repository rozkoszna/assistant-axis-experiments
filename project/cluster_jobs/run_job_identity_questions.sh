#!/usr/bin/env bash
#
# run_job_identity_questions.sh — RunAI batch job: identity-probe intents × user_plausible_traits_strict.
#
# WHY:
#   This experiment measures how a model's responses shift along personality axes depending
#   on the user traits present in the conversation context. The intent file contains
#   identity-probing questions (e.g. "Do you think you're creative?") — questions that
#   ask the model about its own character. We want to see whether and how the model's
#   responses project differently onto personality axes when different user traits are set.
#
# WHERE:
#   Cluster:      RCP production (runai-rcp-prod), project dlab-rozkosz
#   Image:        nvcr.io/nvidia/pytorch:25.05-py3
#   Repo:         /home/rozkosz/persona/assistant-axis  (inside the home PVC)
#   Intent file:  data/identity_probe_intents.jsonl     — prompts probing model self-identity
#   Traits file:  data/axis_trait_lists/user_plausible_traits_strict.json — user trait conditions
#   Axes dir:     precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs
#                 (quality-filtered axes: kept only if ≥10 prompt pairs with ≥3/5 agreement)
#   Output:       written under comparison-name "identity_probe_all_axes_llama_v2" inside the repo
#
# HOW TO RUN (from your laptop, requires runai CLI + OPENAI_API_KEY and HF_TOKEN exported):
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash run_job_identity_questions.sh
#   Monitor: runai describe job identity-run -p dlab-rozkosz
#   Logs:    runai logs identity-run -p dlab-rozkosz
#
# KEY PARAMETERS:
#   --intents-file        What kinds of prompts the model receives (identity questions here).
#   --num-candidates 12   Generate 12 response candidates per prompt.
#   --top-k 4             Keep the 4 highest-scoring candidates (scored by judge model).
#   --temperature 0.8     Sampling temperature; high enough to get diverse candidates.
#   --projection-mode all Project responses onto every available axis (not just top-N).
#   --judge-model         gpt-4.1-mini scores candidate responses (needs OPENAI_API_KEY).
#   --generation-model    Llama-3.1-8B-Instruct generates responses (needs HF_TOKEN).

set -euo pipefail

runai-rcp-prod delete job identity-run -p dlab-rozkosz || true

runai-rcp-prod submit identity-run \
  --image nvcr.io/nvidia/pytorch:25.05-py3 \
  --gpu 1 \
  --environment HOME="/home/rozkosz" \
  --environment USER=rozkosz \
  --environment LOGNAME=rozkosz \
  --environment OPENAI_API_KEY="$OPENAI_API_KEY" \
  --environment HF_TOKEN="$HF_TOKEN" \
  --run-as-uid 264459 \
  --run-as-gid 30154 \
  --supplemental-groups 60220 \
  --existing-pvc claimname=dlab-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/rozkosz \
  --command -- /bin/bash -lc '
    set -euo pipefail
    cd /home/rozkosz/persona/assistant-axis
    source .venv/bin/activate
    export PATH=$HOME/.local/bin:$PATH
    uv run python project/runners/run_multi_trait_analysis.py --user-traits-file data/axis_trait_lists/user_plausible_traits_strict.json --comparison-name identity_probe_all_axes_llama_v2 --intents-file data/identity_probe_intents.jsonl --num-candidates 12 --selection-mode top_k --top-k 4 --generation-model meta-llama/Llama-3.1-8B-Instruct --judge-model gpt-4.1-mini --projection-model meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs --projection-mode all --min-selected 30
  ' \
  -p dlab-rozkosz
