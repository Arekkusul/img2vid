#!/usr/bin/env bash
# Real end-to-end check: real model, real generation, real wall-clock time.
# Not part of `pytest` — this hits the converted LTX-2.5 weights and takes minutes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

MODEL="${IMG2VID_LTX_MODEL:-$HOME/.cache/img2vid/ltx23-model}"
PROMPT="${1:-a heavy wooden door creaks slowly open}"
OUTPUT="outputs/verify.mp4"

test -d "$MODEL" || { echo "FAIL: $MODEL not found; run img2vid-ltx-convert first" >&2; exit 1; }

echo "--- generating (text-to-video, this takes minutes) ---"
time "$VENV_BIN/img2vid" --model "$MODEL" --prompt "$PROMPT" --output "$OUTPUT" --seed 42

echo "--- structural checks ---"
test -s "$OUTPUT" || { echo "FAIL: $OUTPUT missing or empty" >&2; exit 1; }
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate -show_entries format=duration \
  -of default=noprint_wrappers=1 "$OUTPUT"

echo "OK: $OUTPUT generated. Open it to visually confirm coherence (not scriptable)."
