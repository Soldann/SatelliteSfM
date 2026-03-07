#!/bin/bash
# Run the full pipeline for all OMA scenes.
# Pass any extra flags through to run_pipeline.sh (e.g. --skip-preprocess, --dry-run).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OMA_SCENES=(
    OMA_026 OMA_042 OMA_059 OMA_064 OMA_070 OMA_084 OMA_108 OMA_124
    OMA_129 OMA_130 OMA_132 OMA_134 OMA_142 OMA_144 OMA_151 OMA_163
    OMA_172 OMA_176 OMA_181 OMA_198 OMA_203 OMA_211 OMA_212 OMA_221
    OMA_225 OMA_230 OMA_244 OMA_247 OMA_248 OMA_251 OMA_258 OMA_269
    OMA_276 OMA_278 OMA_281 OMA_287 OMA_288 OMA_292 OMA_315 OMA_329
    OMA_331 OMA_332 OMA_342 OMA_353 OMA_355 OMA_357 OMA_364 OMA_367
    OMA_374 OMA_376 OMA_381 OMA_382 OMA_383 OMA_389 OMA_391
)

exec "$SCRIPT_DIR/run_pipeline.sh" "$@" "${OMA_SCENES[@]}"
