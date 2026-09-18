#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../krea-gen"
cargo build --release
echo "Built: $(pwd)/target/release/krea-gen"
