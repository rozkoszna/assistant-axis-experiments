#!/usr/bin/env bash
# Non-interactive RunAI job: user_prompt_intents x user_plausible_traits_strict

set -euo pipefail

runai-rcp-prod submit user-prompts-run \
  --image nvcr.io/nvidia/pytorch:25.05-py3 \
  --gpu 1 \
  --environment HOME="/home/rozkosz" \
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
    export HF_HOME=/home/rozkosz/.cache/huggingface
    mkdir -p "$HF_HOME"
    unset TRANSFORMERS_CACHE
    hf auth login --token "$HF_TOKEN"
    hf auth whoami

    uv run python project/run_multi_trait_analysis.py \
      --reuse-models \
      --user-traits-file data/axis_trait_lists/user_plausible_traits_strict.json \
      --comparison-name strict_all_axes_llama_100eval_v2 \
      --intents-file data/user_prompt_intents.jsonl \
      --num-candidates 12 \
      --selection-mode top_k \
      --top-k 4 \
      --generation-model meta-llama/Llama-3.1-8B-Instruct \
      --judge-model gpt-4.1-mini \
      --projection-model meta-llama/Llama-3.1-8B-Instruct \
      --temperature 0.8 \
      --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs \
      --projection-mode all
  ' \
  -p dlab-rozkosz
