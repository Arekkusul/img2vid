#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh

/opt/homebrew/bin/python3.13 -m venv .venv --without-pip
"$VENV_PYTHON" -m ensurepip --upgrade --default-pip
"$VENV_BIN/pip" install --upgrade pip
"$VENV_BIN/pip" install -e ".[dev]"

echo "Setup complete. Activate with: source scripts/env.sh && source .venv/bin/activate"
