#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

MODEL="${IMG2VID_MODEL:-AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit}"
REQUIRED_GB=20

AVAIL_GB=$(df -g / | tail -1 | awk '{print $4}')
if [ "$AVAIL_GB" -lt "$REQUIRED_GB" ]; then
  echo "Only ${AVAIL_GB}GB free; need >= ${REQUIRED_GB}GB before downloading the ${MODEL} package." >&2
  exit 1
fi

"$VENV_BIN/mlxgen" download --model "$MODEL"

echo "--- capabilities check ---"
"$VENV_BIN/mlxgen" capabilities --model "$MODEL"

echo "--- disk after download ---"
df -h /
du -sh "$HOME/.cache/huggingface/hub" 2>/dev/null || true
