#!/usr/bin/env bash
#
# run_job_opinion_questions.sh — RunAI batch job: opinion-prompt intents × user_plausible_traits_strict.
#
# WHY:
#   Same axis-projection analysis as the identity and user-prompt jobs, but using opinion
#   prompts as the intent type (e.g. "What do you think about X?"). This lets us compare
#   whether the model's response shifts along personality axes differ across intent types:
#   does the model adapt differently when asked for opinions vs. when asked about itself
#   or given a realistic user request?
#
# WHERE:
#   Cluster:      RCP production (runai-rcp-prod), project dlab-rozkosz
#   Image:        nvcr.io/nvidia/pytorch:25.05-py3
#   Repo:         /home/rozkosz/persona/assistant-axis  (inside the home PVC)
#   Intent file:  data/opinion_prompt_intents.jsonl     — prompts asking for the model's opinions
#   Traits file:  data/axis_trait_lists/user_plausible_traits_strict.json — user trait conditions
#   Axes dir:     precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs
#                 (quality-filtered axes: kept only if ≥10 prompt pairs with ≥3/5 agreement)
#   Output:       written under comparison-name "opinion_all_axes_llama" inside the repo
#
# HOW TO RUN (from your laptop, requires runai CLI + OPENAI_API_KEY and HF_TOKEN exported):
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash run_job_opinion_questions.sh
#   Monitor: runai describe job opinion-run -p dlab-rozkosz
#   Logs:    runai logs opinion-run -p dlab-rozkosz
#
# KEY PARAMETERS:
#   --intents-file        What kinds of prompts the model receives (opinion questions here).
#   --num-candidates 12   Generate 12 response candidates per prompt.
#   --top-k 4             Keep the 4 highest-scoring candidates (scored by judge model).
#   --temperature 0.8     Sampling temperature; high enough to get diverse candidates.
#   --projection-mode all Project responses onto every available axis (not just top-N).
#   --judge-model         gpt-4.1-mini scores candidate responses (needs OPENAI_API_KEY).
#   --generation-model    Llama-3.1-8B-Instruct generates responses (needs HF_TOKEN).

set -euo pipefail

runai-rcp-prod delete job opinion-run -p dlab-rozkosz || true

runai-rcp-prod submit opinion-run \
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
    export XDG_CACHE_HOME=/scratch/rozkosz/.cache
    export UV_CACHE_DIR=/scratch/rozkosz/.cache/uv
    mkdir -p /scratch/rozkosz/.cache/uv
    uv run python project/runners/run_multi_trait_analysis.py --user-traits-file data/axis_trait_lists/user_plausible_traits_strict.json --comparison-name opinion_all_axes_llama --intents-file data/opinion_prompt_intents.jsonl --num-candidates 12 --selection-mode top_k --top-k 4 --generation-model meta-llama/Llama-3.1-8B-Instruct --judge-model gpt-4.1-mini --projection-model meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs --projection-mode all
  ' \
  -p dlab-rozkosz
