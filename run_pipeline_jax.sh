#!/bin/bash
# Run the full pipeline for all JAX scenes.
# Pass any extra flags through to run_pipeline.sh (e.g. --skip-preprocess, --dry-run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JAX_SCENES=(
    JAX_004 JAX_017 JAX_018 JAX_020 JAX_022 JAX_028 JAX_031 JAX_033
    JAX_068 JAX_070 JAX_072 JAX_079 JAX_105 JAX_113 JAX_117 JAX_118
    JAX_122 JAX_144 JAX_149 JAX_156 JAX_159 JAX_161 JAX_164 JAX_165
    JAX_166 JAX_167 JAX_168 JAX_175 JAX_178 JAX_179 JAX_203 JAX_204
    JAX_207 JAX_214 JAX_224 JAX_236 JAX_240 JAX_251 JAX_260 JAX_264
    JAX_269 JAX_273 JAX_276 JAX_280 JAX_314 JAX_359 JAX_412 JAX_416
    JAX_427 JAX_467 JAX_474 JAX_505 JAX_559
)

exec "$SCRIPT_DIR/run_pipeline.sh" "$@" "${JAX_SCENES[@]}"
