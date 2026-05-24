#!/usr/bin/env bash
#
# run_job_natural_questions.sh — RunAI batch job: natural-question intents × user_plausible_traits_strict.
#
# WHY:
#   Same axis-projection analysis as the identity and opinion jobs, but using natural
#   everyday questions as the intent type (not self-identity or opinion questions).
#   This is the most naturalistic condition: we measure whether the model's responses shift
#   along personality axes in response to user traits even for ordinary task-like prompts.
#   The comparison name "strict_all_axes_llama_100eval_v2" reflects the strict trait filter
#   and v2 of the evaluation (100-sample eval set).
#
# WHERE:
#   Cluster:      RCP production (runai-rcp-prod), project dlab-rozkosz
#   Image:        nvcr.io/nvidia/pytorch:25.05-py3
#   Repo:         /home/rozkosz/persona/assistant-axis  (inside the home PVC)
#   Intent file:  data/user_prompt_intents.jsonl        — realistic everyday user questions
#   Traits file:  data/axis_trait_lists/user_plausible_traits_strict.json — user trait conditions
#   Axes dir:     precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs
#                 (quality-filtered axes: kept only if ≥10 prompt pairs with ≥3/5 agreement)
#   Output:       written under comparison-name "strict_all_axes_llama_100eval_v2" inside the repo
#
# HOW TO RUN (from your laptop, requires runai CLI + OPENAI_API_KEY and HF_TOKEN exported):
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash run_job_natural_questions.sh
#   Monitor: runai describe job user-prompts-run -p dlab-rozkosz
#   Logs:    runai logs user-prompts-run -p dlab-rozkosz
#
# KEY PARAMETERS:
#   --intents-file        What kinds of prompts the model receives (natural everyday questions here).
#   --num-candidates 12   Generate 12 response candidates per prompt.
#   --top-k 4             Keep the 4 highest-scoring candidates (scored by judge model).
#   --temperature 0.8     Sampling temperature; high enough to get diverse candidates.
#   --projection-mode all Project responses onto every available axis (not just top-N).
#   --judge-model         gpt-4.1-mini scores candidate responses (needs OPENAI_API_KEY).
#   --generation-model    Llama-3.1-8B-Instruct generates responses (needs HF_TOKEN).

set -euo pipefail

runai-rcp-prod delete job user-prompts-run -p dlab-rozkosz || true

runai-rcp-prod submit user-prompts-run \
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
    uv run python project/runners/run_multi_trait_analysis.py --user-traits-file data/axis_trait_lists/user_plausible_traits_strict.json --comparison-name strict_all_axes_llama_100eval_v2 --intents-file data/user_prompt_intents.jsonl --num-candidates 12 --selection-mode top_k --top-k 4 --generation-model meta-llama/Llama-3.1-8B-Instruct --judge-model gpt-4.1-mini --projection-model meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs --projection-mode all
  ' \
  -p dlab-rozkosz
