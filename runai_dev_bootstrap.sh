#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   OPENAI_API_KEY=... bash runai_dev_bootstrap.sh
# If OPENAI_API_KEY is not exported, the fallback below is used.
OPENAI_API_KEY_DEFAULT="${OPENAI_API_KEY:-xxx}"

source ~/.runai_aliases

runai delete job dev -p dlab-rozkosz || true

runai-rcp-prod submit dev \
  --image nvcr.io/nvidia/pytorch:25.05-py3 \
  --gpu 1 \
  --environment HOME="/home/rozkosz" \
  --run-as-uid 264459 \
  --run-as-gid 30154 \
  --supplemental-groups 60220 \
  --existing-pvc claimname=dlab-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/rozkosz \
  --interactive \
  --command \
  -- /bin/bash -ic "sleep infinity" \
  -p dlab-rozkosz

echo "Waiting for job 'dev' to become Running..."
max_tries=60
sleep_seconds=5
for ((i=1; i<=max_tries; i++)); do
  if runai describe job dev -p dlab-rozkosz | grep -qi "Running"; then
    echo "Job is Running."
    break
  fi
  if (( i == max_tries )); then
    echo "Timed out waiting for job 'dev' to become Running."
    echo "Check with: runai describe job dev -p dlab-rozkosz"
    exit 1
  fi
  sleep "$sleep_seconds"
done

runai-rcp-prod bash dev -p dlab-rozkosz <<EOF
kubectl get pods
cd ~/persona/assistant-axis
source .venv/bin/activate
export PATH=\$HOME/.local/bin:\$PATH
export OPENAI_API_KEY="${OPENAI_API_KEY_DEFAULT}"
export USER=rozkosz
export LOGNAME=rozkosz
export HOME=/home/rozkosz
echo "Environment ready in ~/persona/assistant-axis"
EOF

