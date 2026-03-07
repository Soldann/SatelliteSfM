#!/bin/bash

# End-to-end pipeline script for satellite imagery processing.
# Orchestrates all four pipeline stages via docker compose run.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
SKIP_PREPROCESS=false
SKIP_PREPARE=false
SKIP_SFM=false
SKIP_POSTPROCESS=false
USE_SRTM4=false
ENABLE_DEBUG=false
SYMLINK=false
DRY_RUN=false

JAX_RGB_DIR="data/Track3-RGB-1"
OMA_RGB_DIR="data/Track3-RGB-2"
DSM_DIR="data/Track3-Truth"
PROCESSED_DIR="data/DFC2019_processed_new" # This is to avoid overwriting the original processed data

DOCKER_SERVICE="satellitesfm"

usage() {
    echo "Usage: $0 [OPTIONS] SCENE [SCENE ...]"
    echo ""
    echo "SCENE format: CITY_SCENEID  (e.g., JAX_004  OMA_068)"
    echo ""
    echo "Pipeline options:"
    echo "  --skip-preprocess     Skip initial TIF->PNG preprocessing (preprocess_track3.py)"
    echo "  --skip-prepare        Skip scene input preparation (prepare_input.py)"
    echo "  --skip-sfm            Skip SatelliteSfM reconstruction (satellite_sfm.py)"
    echo "  --skip-postprocess    Skip post-processing (postprocess_datasets.sh)"
    echo ""
    echo "SfM options:"
    echo "  --use-srtm4           Use SRTM4 elevation data (recommended)"
    echo "  --enable-debug        Enable SfM debug visualizations"
    echo ""
    echo "Prepare options:"
    echo "  --symlink             Use symlinks instead of copying images"
    echo ""
    echo "Data paths (defaults shown):"
    echo "  --jax-rgb-dir  DIR    $JAX_RGB_DIR"
    echo "  --oma-rgb-dir  DIR    $OMA_RGB_DIR"
    echo "  --dsm-dir      DIR    $DSM_DIR"
    echo "  --processed-dir DIR   $PROCESSED_DIR"
    echo ""
    echo "Other:"
    echo "  --dry-run             Print commands without executing"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --use-srtm4 JAX_004"
    echo "  $0 --skip-preprocess --use-srtm4 JAX_004 OMA_068 JAX_214"
    echo "  $0 --skip-preprocess --skip-prepare --skip-sfm JAX_004"
    echo "  $0 --dry-run --use-srtm4 JAX_004"
}

# Parse arguments
scenes=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-prepare)
            SKIP_PREPARE=true
            shift
            ;;
        --skip-sfm)
            SKIP_SFM=true
            shift
            ;;
        --skip-postprocess)
            SKIP_POSTPROCESS=true
            shift
            ;;
        --use-srtm4)
            USE_SRTM4=true
            shift
            ;;
        --enable-debug)
            ENABLE_DEBUG=true
            shift
            ;;
        --symlink)
            SYMLINK=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --jax-rgb-dir)
            JAX_RGB_DIR="$2"
            shift 2
            ;;
        --oma-rgb-dir)
            OMA_RGB_DIR="$2"
            shift 2
            ;;
        --dsm-dir)
            DSM_DIR="$2"
            shift 2
            ;;
        --processed-dir)
            PROCESSED_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo -e "${RED}Error: Unknown option $1${NC}" >&2
            usage
            exit 1
            ;;
        *)
            scenes+=("$1")
            shift
            ;;
    esac
done

if [ ${#scenes[@]} -eq 0 ]; then
    echo -e "${RED}Error: No scenes specified.${NC}" >&2
    usage
    exit 1
fi

# Validate scene format and collect unique cities
declare -A city_seen
cities_ordered=()
declare -A scene_city
declare -A scene_id

for scene in "${scenes[@]}"; do
    if [[ ! $scene =~ ^([A-Z]+)_([0-9]+)$ ]]; then
        echo -e "${RED}Error: Invalid scene format '$scene'. Expected CITY_SCENEID (e.g., JAX_004).${NC}" >&2
        exit 1
    fi
    city="${BASH_REMATCH[1]}"
    sid="${BASH_REMATCH[2]}"
    scene_city[$scene]="$city"
    scene_id[$scene]="$sid"
    if [[ -z "${city_seen[$city]+x}" ]]; then
        city_seen[$city]=1
        cities_ordered+=("$city")
    fi
done

# Helper: run or dry-run a command
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} $*"
    else
        "$@"
    fi
}

# Helper: run a command inside the container
docker_run() {
    run_cmd docker compose run --rm "$DOCKER_SERVICE" "$@"
}

# Tracking
success_scenes=()
failed_scenes=()

echo ""
echo -e "${BLUE}=========================================="
echo "  SatelliteSfM End-to-End Pipeline"
echo -e "==========================================${NC}"
echo -e "${GREEN}Scenes: ${scenes[*]}${NC}"
echo ""

# ── Step 1: preprocess_track3 (once per unique city) ────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    echo -e "${BLUE}-- Step 1: TIF Preprocessing (per city) --${NC}"
    for city in "${cities_ordered[@]}"; do
        echo ""
        echo -e "${GREEN}Preprocessing city: ${city}${NC}"

        case "$city" in
            JAX) rgb_dir="$JAX_RGB_DIR" ;;
            OMA) rgb_dir="$OMA_RGB_DIR" ;;
            *)
                echo -e "${YELLOW}Warning: Unknown city '$city' — skipping preprocess. Set --jax-rgb-dir / --oma-rgb-dir or add support manually.${NC}"
                continue
                ;;
        esac

        out_dir="data/DFC2019_${city}_preprocessed"
        docker_run python preprocess_track3/preprocess_track3.py \
            --base_view_dir "$rgb_dir" \
            --base_dsm_dir "$DSM_DIR" \
            --out_dir "$out_dir"
    done
else
    echo -e "${YELLOW}-- Step 1: Skipping TIF preprocessing --${NC}"
fi

# ── Step 2: prepare_input (once per scene) ──────────────────────────────────
if [ "$SKIP_PREPARE" = false ]; then
    echo ""
    echo -e "${BLUE}-- Step 2: Scene Input Preparation --${NC}"
    for scene in "${scenes[@]}"; do
        echo ""
        echo -e "${GREEN}Preparing scene: ${scene}${NC}"
        city="${scene_city[$scene]}"
        sid="${scene_id[$scene]}"

        case "$city" in
            JAX) rgb_dir="$JAX_RGB_DIR" ;;
            OMA) rgb_dir="$OMA_RGB_DIR" ;;
            *) rgb_dir="$JAX_RGB_DIR" ;;
        esac

        prepare_args=(python prepare_input.py
            --scene_id "$sid"
            --city "$city"
            --track_rgb_dir "$rgb_dir"
            --output_dir "$PROCESSED_DIR")
        if [ "$SYMLINK" = true ]; then
            prepare_args+=(--symlink)
        fi

        if ! docker_run "${prepare_args[@]}"; then
            echo -e "${RED}Error: prepare_input.py failed for scene ${scene}${NC}"
            failed_scenes+=("$scene (prepare)")
        fi
    done
else
    echo -e "${YELLOW}-- Step 2: Skipping scene preparation --${NC}"
fi

# ── Step 3: satellite_sfm (once per scene) ──────────────────────────────────
if [ "$SKIP_SFM" = false ]; then
    echo ""
    echo -e "${BLUE}-- Step 3: SatelliteSfM Reconstruction --${NC}"
    for scene in "${scenes[@]}"; do
        echo ""
        echo -e "${GREEN}Running SfM for scene: ${scene}${NC}"

        sfm_args=(python satellite_sfm.py \
            --input_folder  "${PROCESSED_DIR}/${scene}/inputs" \
            --output_folder "${PROCESSED_DIR}/${scene}/outputs_srtm" \
            --run_sfm)
        if [ "$USE_SRTM4" = true ];    then sfm_args+=(--use_srtm4);    fi
        if [ "$ENABLE_DEBUG" = true ]; then sfm_args+=(--enable_debug); fi

        if ! docker_run "${sfm_args[@]}"; then
            echo -e "${RED}Error: satellite_sfm.py failed for scene ${scene}${NC}"
            failed_scenes+=("$scene (sfm)")
        fi
    done
else
    echo -e "${YELLOW}-- Step 3: Skipping SfM reconstruction --${NC}"
fi

# ── Step 4: postprocess_datasets (all scenes in one call) ───────────────────
if [ "$SKIP_POSTPROCESS" = false ]; then
    echo ""
    echo -e "${BLUE}-- Step 4: Post-processing --${NC}"
    echo -e "${GREEN}Scenes: ${scenes[*]}${NC}"

    post_args=(bash postprocess_datasets.sh --base-dir "$PROCESSED_DIR")
    if [ "$DRY_RUN" = true ]; then post_args+=(--dry-run); fi
    post_args+=("${scenes[@]}")

    if ! docker_run "${post_args[@]}"; then
        echo -e "${RED}Error: postprocess_datasets.sh failed${NC}"
        for scene in "${scenes[@]}"; do
            failed_scenes+=("$scene (postprocess)")
        done
    else
        for scene in "${scenes[@]}"; do
            success_scenes+=("$scene")
        done
    fi
else
    echo -e "${YELLOW}-- Step 4: Skipping post-processing --${NC}"
    for scene in "${scenes[@]}"; do
        success_scenes+=("$scene")
    done
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLUE}=========================================="
echo "  Pipeline Summary"
echo -e "==========================================${NC}"
echo -e "${GREEN}Scenes requested: ${#scenes[@]}${NC}"
echo -e "${RED}Failures: ${#failed_scenes[@]}${NC}"

if [ ${#failed_scenes[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed steps:${NC}"
    for entry in "${failed_scenes[@]}"; do
        echo "  - $entry"
    done
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}All pipeline steps completed successfully.${NC}"
echo ""
