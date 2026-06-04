#!/usr/bin/env bash
#
# launch_parallel_traits.sh — split the 50 traits across N GPUs as N parallel RunAI jobs.
#
# WHY:
#   The single-GPU runs process 50 traits sequentially (~hours each → up to ~2 days).
#   Every trait writes to its own subfolder outputs/<comparison-name>/<trait>/..., so traits
#   are fully independent — there is no cross-trait state. We can therefore fan them out across
#   the cluster's GPUs: split 50 traits into N chunks and submit N jobs, all sharing the same
#   --comparison-name. Wall-clock time drops ~N-fold. The subprocess pipeline skips any trait
#   whose projection file already exists, so re-running is safe and resumable.
#
# HOW TO RUN (from your laptop, requires runai CLI + OPENAI_API_KEY and HF_TOKEN exported):
#   COMPARISON_NAME=strict_all_axes_llama_100eval_v2_label_only \
#   EXTRA_ARGS="--explicit-label-neutral" \
#   NUM_JOBS=10 \
#   OPENAI_API_KEY=sk-... HF_TOKEN=hf_... bash launch_parallel_traits.sh
#
#   Monitor all jobs:  runai list jobs -p dlab-rozkosz | grep user-prompts-par
#   Logs for chunk 3:  runai logs user-prompts-par-3 -p dlab-rozkosz
#   Delete all:        for i in $(seq 0 $((NUM_JOBS-1))); do runai-rcp-prod delete job user-prompts-par-$i -p dlab-rozkosz; done
#
# ENV VARS:
#   COMPARISON_NAME  (required) output dir name, e.g. strict_all_axes_llama_100eval_v2_label_only
#   INTENTS_FILE     (default data/user_prompt_intents.jsonl)
#   TRAITS_FILE      (default data/axis_trait_lists/user_plausible_traits_strict.json)
#   EXTRA_ARGS       (optional) extra flags passed to run_multi_trait_analysis.py, e.g. "--explicit-label-neutral"
#   NUM_JOBS         (default 10) number of parallel jobs / GPUs to use
#   JOB_PREFIX       (default user-prompts-par) RunAI job name prefix

set -euo pipefail

COMPARISON_NAME="${COMPARISON_NAME:?Set COMPARISON_NAME}"
INTENTS_FILE="${INTENTS_FILE:-data/user_prompt_intents.jsonl}"
TRAITS_FILE="${TRAITS_FILE:-data/axis_trait_lists/user_plausible_traits_strict.json}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NUM_JOBS="${NUM_JOBS:-10}"
JOB_PREFIX="${JOB_PREFIX:-user-prompts-par}"

# Read all traits from the JSON list (laptop-side; the file is in the repo).
ALL_TRAITS=($(python3 -c "import json,sys; print(' '.join(json.load(open('${TRAITS_FILE}'))))"))
TOTAL=${#ALL_TRAITS[@]}
echo "Splitting ${TOTAL} traits across ${NUM_JOBS} jobs (comparison=${COMPARISON_NAME})"

# Ceil division so the last chunk picks up any remainder.
CHUNK=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))

for (( j=0; j<NUM_JOBS; j++ )); do
  START=$(( j * CHUNK ))
  if (( START >= TOTAL )); then
    break  # fewer non-empty chunks than NUM_JOBS (e.g. NUM_JOBS > TOTAL)
  fi
  SUBSET=("${ALL_TRAITS[@]:START:CHUNK}")
  JOB_NAME="${JOB_PREFIX}-${j}"
  echo "  ${JOB_NAME}: ${SUBSET[*]}"

  runai-rcp-prod delete job "${JOB_NAME}" -p dlab-rozkosz || true

  runai-rcp-prod submit "${JOB_NAME}" \
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
    --command -- /bin/bash -lc "
      set -euo pipefail
      cd /home/rozkosz/persona/assistant-axis
      source .venv/bin/activate
      export PATH=\$HOME/.local/bin:\$PATH
      export XDG_CACHE_HOME=/tmp/cache
      export UV_CACHE_DIR=/tmp/cache/uv
      mkdir -p /tmp/cache/uv
      uv run python project/runners/run_multi_trait_analysis.py --user-traits ${SUBSET[*]} --comparison-name ${COMPARISON_NAME} --intents-file ${INTENTS_FILE} --num-candidates 12 --selection-mode top_k --top-k 4 --generation-model meta-llama/Llama-3.1-8B-Instruct --judge-model gpt-4.1-mini --projection-model meta-llama/Llama-3.1-8B-Instruct --temperature 0.8 --axes-dir precomputed_axis/answer_mean/filter_prompt_pair_question_wins_ge_10_require_3_of_5_prompt_pairs --projection-mode all ${EXTRA_ARGS}
    " \
    -p dlab-rozkosz
done

echo "Submitted ${NUM_JOBS} parallel jobs. Monitor: runai list jobs -p dlab-rozkosz | grep ${JOB_PREFIX}"
