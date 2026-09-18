#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

REPO="${IMG2VID_DISTILL_LORA_REPO:-Arekkusul/krea-2-raw-distill-lora}"
OUTPUT_FILE="${IMG2VID_DISTILL_LORA_PATH:-$HOME/.cache/img2vid/krea2-distill-lora.safetensors}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
"$VENV_BIN/hf" download "$REPO" krea2_distill_x2.safetensors --local-dir /tmp/img2vid-distill-lora-dl
mv /tmp/img2vid-distill-lora-dl/krea2_distill_x2.safetensors "$OUTPUT_FILE"
rm -rf /tmp/img2vid-distill-lora-dl

echo "--- downloaded to $OUTPUT_FILE ---"
du -sh "$OUTPUT_FILE"
