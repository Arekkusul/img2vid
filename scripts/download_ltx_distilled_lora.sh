#!/usr/bin/env bash
# Fetches the official Lightricks distilled LoRA (the fusion INPUT for
# scripts/fuse_distilled_lora.sh -- not to be confused with this project's own
# already-fused output, transformer-distilled.safetensors).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

REPO="${IMG2VID_LTX_LORA_REPO:-Lightricks/LTX-2.5}"
FILE="loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
OUTPUT_FILE="${IMG2VID_LTX_LORA_PATH:-$HOME/.cache/img2vid/ltx-2.5-distilled-lora-450-bf16.safetensors}"

if ! "$VENV_BIN/hf" auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Run: $VENV_BIN/hf auth login" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
if ! "$VENV_BIN/hf" download "$REPO" "$FILE" --local-dir /tmp/img2vid-ltx-lora-dl; then
  echo "" >&2
  echo "Download failed. $REPO is a gated repo -- being logged in isn't enough, this" >&2
  echo "specific repo also needs its own access request approved:" >&2
  echo "  1. Visit https://huggingface.co/$REPO and request/accept access" >&2
  echo "  2. Wait for approval (may not be instant), then re-run this script" >&2
  rm -rf /tmp/img2vid-ltx-lora-dl
  exit 1
fi
mv "/tmp/img2vid-ltx-lora-dl/$FILE" "$OUTPUT_FILE"
rm -rf /tmp/img2vid-ltx-lora-dl

echo "--- downloaded to $OUTPUT_FILE ---"
du -sh "$OUTPUT_FILE"
echo "--- now fuse it onto your own dev transformer: ---"
echo "scripts/fuse_distilled_lora.sh --lora $OUTPUT_FILE"
