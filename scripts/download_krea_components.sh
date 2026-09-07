#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

MODEL="${IMG2VID_KREA_MODEL:-krea/Krea-2-Raw}"
OUTPUT_DIR="${IMG2VID_KREA_SNAPSHOT:-$HOME/.cache/img2vid/krea2-snapshot}"

if ! "$VENV_BIN/hf" auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. $MODEL is a gated repo:" >&2
  echo "  1. Visit https://huggingface.co/$MODEL and accept the license" >&2
  echo "  2. Run: $VENV_BIN/hf auth login" >&2
  exit 1
fi

"$VENV_BIN/hf" download "$MODEL" \
  --include "text_encoder/*" \
  --include "vae/*" \
  --include "tokenizer/*" \
  --include "scheduler/*" \
  --include "model_index.json" \
  --local-dir "$OUTPUT_DIR"

echo "--- downloaded to $OUTPUT_DIR ---"
du -sh "$OUTPUT_DIR"/{text_encoder,vae} 2>/dev/null || true
