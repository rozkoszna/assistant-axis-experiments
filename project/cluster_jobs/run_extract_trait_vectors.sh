#!/usr/bin/env bash
#
# run_extract_trait_vectors.sh — Extract hidden-state trait vectors from Llama-3.1-8B on the cluster.
#
# WHY:
#   To build "personality axes" we need to know how the model internally represents
#   trait-related concepts. This script runs extract_trait_vectors.py, which feeds
#   trait-probing questions through Llama-3.1-8B, captures the residual-stream
#   activations at a chosen layer (default: 16), and saves a direction vector per trait.
#   Those vectors are later used as projection axes in the multi-trait analysis.
#
# WHERE:
#   This script is designed to run INSIDE a RunAI job on the RCP/IC cluster, not locally.
#   It must be copied to scratch first because the cluster nodes do not have access to
#   your laptop's filesystem.
#   - Scratch (large, fast):  /dlabscratch1/<user>/  (or /mnt/... depending on mount)
#   - Repo on scratch:        $BASE/assistant-axis-llama3.1-8B/
#   - Output vectors:         $BASE/assistant_axis_outputs/llama-3.1-8b/vectors_q50/
#   - HF model cache:         $BASE/.cache/huggingface/  (redirected away from home quota)
#
# HOW TO RUN:
#   1. Copy this file to scratch:
#        cp run_extract_trait_vectors.sh /mnt/dlabscratch1/bazina/assistant-axis-llama3.1-8B/
#   2. Fill in OPENAI_API_KEY below (needed for the GPT-4o-mini judge).
#   3. Submit a RunAI job that executes it:
#        rs extract-trait-vectors --gpu 1.0 --memory 40G --node-type G10 -- \
#          /dlabscratch1/bazina/assistant-axis-llama3.1-8B/run_extract_trait_vectors.sh
#
# KEY PARAMETERS:
#   --trait_dir   Directory of per-trait question files (JSONL), one file per trait.
#   --model_id    HuggingFace model to extract from. Layer count must match --layer.
#   --layer       Residual-stream layer to read activations from (16 = mid-network for 8B).
#   --out_dir     Where to write the .npy vector files, one per trait.
#   --judge_model GPT model used to score/filter extraction questions (needs OPENAI_API_KEY).

set -euo pipefail
: "${USER:=$(whoami)}"

if   [ -d /dlabscratch1/"$USER" ];                  then BASE=/dlabscratch1/"$USER"
elif [ -d /mnt/dlabscratch1/"$USER" ];              then BASE=/mnt/dlabscratch1/"$USER"
elif [ -d /mnt/dlab/scratch/dlabscratch1/"$USER" ]; then BASE=/mnt/dlab/scratch/dlabscratch1/"$USER"
else echo "ERROR: scratch not found"; exit 1; fi

REPO="assistant-axis-llama3.1-8B"
OUTPUTS="assistant_axis_outputs/llama-3.1-8b"
OPENAI_API_KEY="YOUR_KEY_HERE"

cd "$BASE/$REPO"

export HF_HOME=$BASE/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$BASE/.cache
export TORCH_HOME=$BASE/.cache/torch
export TMPDIR=$BASE/.tmp
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

uv sync

uv run tools/extract_trait_vectors.py \
    --trait_dir      "$BASE/$REPO/data/extraction_questions/traits" \
    --model_id       meta-llama/Llama-3.1-8B-Instruct \
    --out_dir        "$BASE/$OUTPUTS/vectors_q50" \
    --layer          16 \
    --openai_api_key "$OPENAI_API_KEY" \
    --judge_model    gpt-4o-mini
