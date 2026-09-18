#!/usr/bin/env bash
# Fuse the LTX-2.5 distilled LoRA onto this project's own converted dev transformer.
# Must run under ltx-2-mlx-upstream's own venv (needs ltx_core_mlx) -- this wrapper
# invokes that venv's python directly, mirroring how generate.py resolves ltx-2-mlx itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LTX_PYTHON="ltx-2-mlx-upstream/.venv/bin/python"
if [ ! -x "$LTX_PYTHON" ]; then
  echo "FAIL: $LTX_PYTHON not found; run: cd ltx-2-mlx-upstream && uv sync" >&2
  exit 1
fi

"$LTX_PYTHON" scripts/fuse_distilled_lora.py "$@"
