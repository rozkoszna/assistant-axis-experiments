#!/usr/bin/env bash
#
# run_job_natural_questions_explicit_prefix.sh — same as natural questions run but with
# an explicit "I am <trait>." prepended to the trait prompt before response generation.
#
# WHY:
#   Ablation comparing implicit trait conditioning (style/tone only) vs. explicit self-
#   disclosure ("I am anxious. [rest of prompt]"). The neutral prompt and underlying request
#   are identical to the strict_all_axes_llama_100eval_v2 run; only the trait prompt changes.
#   This tests whether the model's axis shifts come from linguistic style cues or from
#   reading a direct trait label.
#
# HOW TO RUN (from your laptop, requires runai CLI + OPENAI_API_KEY and HF_TOKEN exported):
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash run_job_natural_questions_explicit_prefix.sh
#   Monitor: runai describe job user-prompts-run-explicit -p dlab-rozkosz
#   Logs:    runai logs user-prompts-run-explicit -p dlab-rozkosz

set -euo pipefail

runai-rcp-prod delete job user-prompts-run-explicit -p dlab-rozkosz || true

runai-rcp-prod submit user-prompts-run-explicit \
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
    export XDG_CACHE_HOME=/tmp/cache
    export UV_CACHE_DIR=/tmp/cache/uv
    mkdir -p /tmp/cache/uv
    uv run python project/runners/run_multi_trait_analysis.py --user-traits-file data/axis_trait_lists/user_plausible_traits_strict.json --comparison-name strict_all_axes_llama_100eval_v2_explicit_prefix --intents-file data/user_prompt_intents.jsonl --num-candidates 12 --selection-mode top_k --top-k 4 --generation-model meta-llama/Llama-3.1-8B-Instruct --judge-model gpt-4.1-mini --projection-model meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs --projection-mode all --explicit-trait-prefix
  ' \
  -p dlab-rozkosz
