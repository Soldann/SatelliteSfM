# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a fork of [SatelliteSfM](https://github.com/Kai-46/SatelliteSfM), extended with additional preprocessing and post-processing tools to prepare satellite imagery datasets for [Skyfall-GS](https://github.com/jayin92/Skyfall-GS) (a 3D Gaussian Splatting model). The pipeline converts satellite `.tif` images with RPC (Rational Polynomial Camera) models into OpenCV-compatible pinhole camera representations via Structure-from-Motion.

## Setup

```bash
# Run from repo root — installs ColmapForVisSat, creates conda env, and installs all deps
. ./env.sh
```

This creates the `SatelliteSfM` conda environment. Always activate it before running any scripts:
```bash
conda activate SatelliteSfM
```

## Complete Pipeline (DFC2019 Satellite Imagery)

The pipeline flows through these stages:

### 1. Preprocess raw TIF images
```bash
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-1 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_JAX_preprocessed
```

### 2. Prepare a single scene's inputs
```bash
python prepare_input.py --scene_id 004 --city JAX [--symlink]
```

Or merge multiple scenes into one SfM run (for a shared coordinate system):
```bash
python merge_scenes.py --scenes 004 005 --city JAX --output_name JAX_merged [--symlink]
```

### 3. Run SatelliteSfM reconstruction
```bash
python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm --use_srtm4 [--enable_debug]
```

### 4. Post-process (skew correction → format conversion → mask generation)
```bash
./postprocess_datasets.sh JAX_004 [OMA_068 ...]
# Options: --skip-skew, --skip-convert, --skip-copy, --skip-mask, --dry-run
```

### 5. COLMAP dataset conversion (alternative path)
```bash
python convert_colmap_datasets.py -s /path/to/colmap_output [--skip_calibration]
```

## Architecture

### Core Processing Flow

**`satellite_sfm.py`** — Entry point; reads `latlonalt_bbx.json`, optionally fetches SRTM4 altitude data, then calls `preprocess_image_set()`.

**`preprocess/preprocess_image_set.py`** — Main orchestrator:
1. Converts `.tif` → `.png` via `parse_tif_image()` (parallel with multiprocessing)
2. Extracts RPC metadata to per-image JSON files in `metas/`
3. Approximates RPC with a local pinhole camera via `approximate_rpc_locally()`, saving `K` + `W2C` matrices to `cameras/`
4. If `--run_sfm`: calls `preprocess_sfm()` for SIFT matching, triangulation, and bundle adjustment
5. Outputs adjusted cameras to `cameras_adjusted/`

**`preprocess_sfm/preprocess_sfm.py`** — SfM pipeline wrapper:
- SIFT matching via ColmapForVisSat
- Iterative triangulation + global bundle adjustment at two reprojection error thresholds (32.0 → 2.0)
- Extracts final cameras to `cameras_adjusted/` and updates `enu_bbx_adjusted.json`

**`preprocess/approximate_rpc_locally.py`** — Converts RPC model to pinhole by sampling points in ENU (East-North-Up) coordinate space and solving for K and W2C.

### Coordinate Systems

- **Global**: latitude/longitude/altitude (WGS84)
- **Local ENU**: East-North-Up, centered on the scene's centroid at min altitude - 20m (the "observer" coordinate)
- **Camera**: OpenCV-compatible pinhole with 4×4 K and W2C matrices (float64 required due to large depth values from satellite altitude)

### Post-processing Scripts

- **`skew_correct.py`**: Removes camera skew (s term in K) via affine warp, updates K and image files in parallel
- **`convert_datasets.py`**: Converts `cameras/` JSON files to `transforms_train.json` / `transforms_test.json` (90/10 split)
- **`generate_masks.py`**: Creates binary masks for non-black pixels (`.npy` + `.png`)
- **`preprocess_rotation.py`**: Computes rotation offsets (0/90/180/270°) between images in a scene using NCC, edge correlation, feature matching, and histogram voting
- **`streamlit_viewer.py`**: Interactive viewer for browsing preprocessed images with rotation

### Key Data Structures

Camera JSON files (`cameras/`, `cameras_adjusted/`):
```json
{"K": [16 floats, 4×4], "W2C": [16 floats, 4×4], "img_size": [width, height]}
```

Scene bounding box (`latlonalt_bbx.json`):
```json
{"lat_minmax": [...], "lon_minmax": [...], "alt_minmax": [...]}
```

ENU bounding box (`enu_bbx_adjusted.json`) additionally contains `u_minmax` (vertical extent) derived from triangulated points.

### ColmapForVisSat

The custom COLMAP build lives at `preprocess_sfm/ColmapForVisSat/` (built during `env.sh`). Commands are called via `preprocess_sfm/colmap_sfm_commands.py` and utilities in `preprocess_sfm/colmap_sfm_utils.py`.

## Data Directory Convention

```
data/
├── Track3-RGB-1/           # Raw JAX .tif images
├── Track3-RGB-2/           # Raw OMA .tif images
├── Track3-Truth/           # Ground truth DSMs
├── DFC2019_JAX_preprocessed/  # After preprocess_track3.py
│   ├── cameras/, images/, metas/, latlonalt_bbx/, enu_bbx/, ...
├── DFC2019_OMA_preprocessed/
└── DFC2019_processed/
    └── JAX_004/
        ├── inputs/images/*.tif + latlonalt_bbx.json
        ├── outputs_srtm/   # SfM output (cameras_adjusted/, colmap_triangulate_postba/, ...)
        └── outputs_skew/   # Final output (cameras/, images/, masks/, transforms_*.json, points3D.txt)
```

## Float64 Precision

Satellite cameras are extremely far from the scene (hundreds of km). All camera math must use **float64** to avoid precision loss. This is a known pitfall when integrating with downstream NeRF/3DGS frameworks that default to float32.
