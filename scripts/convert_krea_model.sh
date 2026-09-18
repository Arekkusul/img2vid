#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

SOURCE="${1:-$HOME/Downloads/imagemodelfp8.safetensors}"
OUTPUT_DIR="${IMG2VID_KREA_SNAPSHOT:-$HOME/.cache/img2vid/krea2-snapshot}"

if [ ! -f "$SOURCE" ]; then
  echo "Source checkpoint not found: $SOURCE" >&2
  exit 1
fi

"$VENV_BIN/img2vid-krea-convert" --source "$SOURCE" --output-dir "$OUTPUT_DIR"
echo "--- converted transformer written to $OUTPUT_DIR/transformer ---"
du -sh "$OUTPUT_DIR/transformer"
