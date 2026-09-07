#!/usr/bin/env bash
# Sourced by every other script in this project. Not meant to be executed directly.
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export VENV_BIN="$PROJECT_ROOT/.venv/bin"
