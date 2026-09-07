#!/usr/bin/env bash
# Real end-to-end check: real model, real generation, real wall-clock time.
# Not part of `pytest` — this hits actual mlxgen weights and takes minutes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

IMAGE="${1:-tests/fixtures/verify_input.jpg}"
PROMPT="${2:-the camera slowly pans across the scene as clouds drift and light shifts}"
OUTPUT="outputs/verify.mp4"

echo "--- capabilities check (weights present?) ---"
"$VENV_BIN/mlxgen" capabilities --model AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit >/dev/null

echo "--- generating (this takes minutes) ---"
time "$VENV_BIN/img2vid" --image "$IMAGE" --prompt "$PROMPT" --output "$OUTPUT" --seed 42

echo "--- structural checks ---"
test -s "$OUTPUT" || { echo "FAIL: $OUTPUT missing or empty" >&2; exit 1; }
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,avg_frame_rate -show_entries format=duration \
  -of default=noprint_wrappers=1 "$OUTPUT"

echo "OK: $OUTPUT generated. Open it to visually confirm coherence (not scriptable)."
