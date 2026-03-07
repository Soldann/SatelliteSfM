# Dataset Preprocessing for Skyfall-GS

This repository is forked from [Kai-46/SatelliteSfM](https://github.com/Kai-46/SatelliteSfM). Special thanks to Kai Zhang for the original work!

It provides tools for preparing datasets for [Skyfall-GS](https://github.com/jayin92/Skyfall-GS) from two sources:

1. **COLMAP reconstructions** — Convert existing COLMAP sparse reconstructions
2. **Satellite imagery** — Process satellite images with RPC camera models using SatelliteSfM

---

## Table of Contents

- [COLMAP Dataset Conversion](#colmap-dataset-conversion)
- [Satellite Imagery Processing](#satellite-imagery-processing)
  - [Environment Setup](#environment-setup)
  - [Download DFC2019 Dataset](#download-dfc2019-dataset)
  - [Pipeline](#pipeline)
    - [1. Initial Preprocessing](#1-initial-preprocessing)
    - [2. Prepare Scene Inputs](#2-prepare-scene-inputs)
    - [3. Run SatelliteSfM](#3-run-satellitesfm)
    - [4. Post-Processing](#4-post-processing)
  - [Complete Workflow Example](#complete-workflow-example)
  - [Batch Processing](#batch-processing)
- [Troubleshooting](#troubleshooting)
- [Background](#background)
- [Citation](#citation)

---

## COLMAP Dataset Conversion

If you already have reconstruction results from [COLMAP](https://github.com/colmap/colmap), convert them with:

```bash
python convert_colmap_datasets.py -s /path/to/colmap_output [--skip_calibration]
```

This script reads the COLMAP sparse model at `<dataset>/sparse/0` and generates `transforms_train.json` and `transforms_test.json`, rotating and centering cameras so that the look-at point is at the scene origin with up vector `(0, 0, 1)`.

> **Dataset ready!** After conversion, your dataset can be used directly with [Skyfall-GS](https://github.com/jayin92/Skyfall-GS).

---

## Satellite Imagery Processing

### What is RPC?

The RPC (Rational Polynomial Camera) model is a standard camera model for satellite imagery. Unlike pinhole cameras, it handles orbital motion, Earth curvature, and atmospheric distortion using rational polynomials to map between `(Lat, Lon, Alt)` world coordinates and `(Row, Col)` pixel coordinates:

```
Row = P1(X,Y,Z) / P2(X,Y,Z)
Col = P3(X,Y,Z) / P4(X,Y,Z)
```

See the [RPC specification](http://geotiff.maptools.org/rpc_prop.html) for details.

---

### Environment Setup

#### Option A: Docker (recommended)

Handles all dependencies (ColmapForVisSat compiled from source, CUDA, GDAL, Python packages) without requiring a specific OS or compiler.

**Requirements:** Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Pre-download the Miniconda installer** (required at build time — network unavailable during Docker build):
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O Miniconda3-installer.sh
```

**Build the image** (~30–45 min, compiles ColmapForVisSat from source):
```bash
docker build -t satellitesfm .
```

**Run interactively:**
```bash
docker compose run satellitesfm
```

#### Option B: Conda (local)

Requires Ubuntu 18.04 (for GCC-7), at least one GPU, and conda installed.

```bash
. ./env.sh
```

---

### Download DFC2019 Dataset

The pipeline uses the [DFC2019 Track 3: Multi-View Semantic Stereo](https://ieee-dataport.org/open-access/data-fusion-contest-2019) benchmark — multi-view satellite imagery of Jacksonville (JAX) and Omaha (OMA).

Download these three archives and extract into `data/`:

| Archive | Size |
|---|---|
| Track 3 / Training data / RGB images 1/2 | 7.6 GB |
| Track 3 / Training data / RGB images 2/2 | 7.6 GB |
| Track 3 / Training data / Reference | 37.5 MB |

Expected structure:
```
data/
├── Track3-RGB-1/*.tif
├── Track3-RGB-2/*.tif
└── Track3-Truth/[*.tif, *.txt]
```

A preprocessed version is also available on [Google Drive](https://drive.google.com/drive/folders/1Do7oF36sCEBWrcIiCzgHbhF5kQMawjVo?usp=sharing).

---

### Pipeline

#### 1. Initial Preprocessing

Convert `.tif` images to PNG and extract RPC metadata:

```bash
# Jacksonville (JAX)
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-1 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_JAX_preprocessed

# Omaha (OMA)
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-2 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_OMA_preprocessed
```

Output:
```
data/DFC2019_[JAX|OMA]_preprocessed/
├── cameras/          # Camera parameters
├── images/           # Converted PNG images
├── metas/            # RPC coefficients and metadata (JSON)
├── enu_bbx/          # ENU coordinate bounding boxes
├── enu_observers/    # Observer positions in ENU coordinates
├── latlonalt_bbx/    # Lat/Lon/Alt bounding boxes
└── groundtruth_u/    # Ground truth data
```

#### 2. Prepare Scene Inputs

Organize images by scene ID:

```bash
# Basic usage
python prepare_input.py --scene_id 004              # JAX scene 004
python prepare_input.py --scene_id 068 --city OMA   # OMA scene 068

# Use symlinks to save disk space
python prepare_input.py --scene_id 004 --symlink
```

Output:
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/inputs/
├── images/
│   └── {CITY}_{SCENE_ID}_*_RGB.tif
└── latlonalt_bbx.json
```

#### 3. Run SatelliteSfM

Perform structure-from-motion reconstruction on the satellite imagery:

```bash
# With Docker
docker compose run satellitesfm python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm \
    --use_srtm4 \
    --enable_debug

# Without Docker
python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm \
    --use_srtm4 \
    --enable_debug
```

| Flag | Description |
|---|---|
| `--run_sfm` | Enable SfM reconstruction |
| `--use_srtm4` | Fetch SRTM4 elevation data for altitude initialization |
| `--enable_debug` | Output visualizations to `outputs_srtm/debug_sfm/` |

Output:
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/outputs_srtm/
├── colmap_triangulate_postba/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.txt
├── cameras_adjusted/   # Bundle-adjusted pinhole cameras (K, W2C matrices)
├── images/             # PNG images
└── enu_bbx_adjusted.json
```

#### 4. Post-Processing

Applies skew correction, converts to Skyfall-GS format, and generates masks:

```bash
chmod +x postprocess_scenes.sh

# Single scene
./postprocess_scenes.sh JAX_004

# Multiple scenes
./postprocess_scenes.sh JAX_004 OMA_068 JAX_214
```

Steps performed for each scene:
1. **Skew correction** — corrects geometric distortion using SRTM elevation data
2. **Format conversion** — converts to Skyfall-GS format (`transforms_train.json`, `transforms_test.json`)
3. **Copy 3D points** — copies `points3D.txt` from SfM output
4. **Mask generation** — creates binary masks for valid (non-black) pixels

Advanced options:
```bash
./postprocess_scenes.sh --skip-skew JAX_004      # Skip skew correction
./postprocess_scenes.sh --skip-convert JAX_004   # Skip format conversion
./postprocess_scenes.sh --skip-copy JAX_004      # Skip points3D.txt copy
./postprocess_scenes.sh --skip-mask JAX_004      # Skip mask generation
./postprocess_scenes.sh --dry-run JAX_004        # Preview without executing
./postprocess_scenes.sh --base-dir /custom JAX_004
```

Final output:
```
data/DFC2019_processed/{CITY}_{SCENE_ID}/outputs_skew/
├── cameras/
├── images/*.png
├── masks/
│   ├── *.npy
│   └── *.png
├── transforms_train.json
├── transforms_test.json
└── points3D.txt
```

> **Dataset ready!** This output can be used directly with [Skyfall-GS](https://github.com/jayin92/Skyfall-GS).

---

### Complete Workflow Example

End-to-end processing of JAX scene 004:

```bash
# 1. Download and extract DFC2019 (manual step)
#    → data/Track3-RGB-1/, data/Track3-RGB-2/, data/Track3-Truth/

# 2. Initial preprocessing
python preprocess_track3/preprocess_track3.py \
    --base_view_dir data/Track3-RGB-1 \
    --base_dsm_dir data/Track3-Truth \
    --out_dir data/DFC2019_JAX_preprocessed

# 3. Prepare scene inputs
python prepare_input.py --scene_id 004 --city JAX

# 4. Run SatelliteSfM (Docker)
docker compose run satellitesfm python satellite_sfm.py \
    --input_folder data/DFC2019_processed/JAX_004/inputs \
    --output_folder data/DFC2019_processed/JAX_004/outputs_srtm \
    --run_sfm --use_srtm4 --enable_debug

# 5. Post-process
./postprocess_scenes.sh JAX_004

# 6. Ready for Skyfall-GS
ls data/DFC2019_processed/JAX_004/outputs_skew/
```

---

### Batch Processing

```bash
scenes=(JAX_004 JAX_068 JAX_214 OMA_001 OMA_175)

# Prepare all inputs
for scene in "${scenes[@]}"; do
    city=$(echo $scene | cut -d'_' -f1)
    scene_id=$(echo $scene | cut -d'_' -f2)
    python prepare_input.py --scene_id $scene_id --city $city --symlink
done

# Run SatelliteSfM for all scenes
for scene in "${scenes[@]}"; do
    docker compose run satellitesfm python satellite_sfm.py \
        --input_folder data/DFC2019_processed/${scene}/inputs \
        --output_folder data/DFC2019_processed/${scene}/outputs_srtm \
        --run_sfm --use_srtm4
done

# Post-process all at once
./postprocess_scenes.sh "${scenes[@]}"
```

---

## Troubleshooting

**"No images found matching pattern"**
- Scene ID must be zero-padded (e.g., `"004"` not `"4"`)
- Verify `.tif` files exist in the Track3-RGB directory

**"Input folder does not exist"**
- Run `prepare_input.py` before `satellite_sfm.py`
- Check scene ID and city code are correct

**SatelliteSfM fails with SRTM errors**
- SRTM data is downloaded at runtime — ensure internet access inside the container
- Fall back to manual altitude: edit `latlonalt_bbx.json` and remove `--use_srtm4`

**COLMAP binary fails to load (`libXXX.so not found`)**
- Should not happen with the Docker image; if it does, run:
  ```bash
  docker run --rm -e LD_LIBRARY_PATH=/app/preprocess_sfm/ColmapForVisSat/build/__install__/lib \
      satellitesfm ldd /app/preprocess_sfm/ColmapForVisSat/build/__install__/bin/colmap | grep "not found"
  ```
  and add the missing package to the runtime apt stage in `Dockerfile`.

**`No module named '_gdal_array'`**
- This means GDAL was compiled before numpy. Rebuild the image — the current `Dockerfile` installs numpy first.

**GPU not detected in container**
- Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed and configured
- Test with: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu18.04 nvidia-smi`

---

## Background

This repo is built on top of [Kai-46/SatelliteSfM](https://github.com/Kai-46/SatelliteSfM), which bridges the gap between computer vision SfM pipelines and remote sensing satellite imagery. The key insight is that satellite images can be processed through standard SfM pipelines if the RPC camera model is first converted to a pinhole model — enabling downstream use of tools like COLMAP, NeRF, and 3DGS.

### Development Roadmap (upstream)

- [x] SatelliteSfM
- [x] [SatelliteNeRF](https://github.com/Kai-46/SatelliteNeRF)
- [x] Camera visualization scripts
- [x] [ColmapForVisSatPatched](https://github.com/SBCV/ColmapForVisSatPatched) (patches for latest COLMAP)
- [x] [SatelliteNeuS](https://github.com/Kai-46/SatelliteNeuS)
- [ ] SatellitePlaneSweep / SatelliteNeRF / SatelliteNeuS documentation
- [ ] Deep Satellite Stereo

### Relevant Repos

- [SatelliteNeRF](https://github.com/Kai-46/SatelliteNeRF) — neural radiance fields for satellite imagery
- [SatelliteNeuS](https://github.com/Kai-46/SatelliteNeuS) — mesh reconstruction from multi-date satellite images
- [ColmapForVisSatPatched](https://github.com/SBCV/ColmapForVisSatPatched) — updated COLMAP fork

### Float32 Pitfall for Downstream Applications

Satellite cameras are far from the scene, producing huge depth values. When using float32 GPU computing (NeRF, MVS, deep stereo), precision loss can cause artifacts. Use `normalize_sfm_reconstruction.py` to center and scale the scene, then adjust `pixel2ray` to use float64 internally:

```python
def pixel2ray(col, row, K, W2C):
    C2W = torch.inverse(W2C)  # float64
    px = torch.stack((col, row, torch.ones_like(col)), axis=-1).unsqueeze(-1).double()
    ray_d = torch.matmul(C2W[:3, :3], torch.matmul(torch.inverse(K[:3, :3]), px))
    ray_d = (ray_d / ray_d.norm(dim=1, keepdims=True)).squeeze(-1)
    ray_o = C2W[:3, 3].unsqueeze(0).expand(ray_d.shape[0], -1)
    shift = torch.norm(ray_o, dim=-1) - 5.
    ray_o = ray_o + ray_d * shift.unsqueeze(-1)
    return ray_o.float(), ray_d.float()
```

### Example Results

**Input images**

![Input images](./readme_resources/example_data.gif)

**Sparse point cloud from SfM**

![Sparse point cloud](./readme_resources/example_data_sfm.gif)

**Camera visualization** (`python visualize_satellite_cameras.py`)

Red/Green/Blue axes = East/North/Up. Each camera is a line from origin to camera center.

![Visualize cameras](./readme_resources/visualize_camera.png)

---

## Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{VisSat-2019,
  title={Leveraging Vision Reconstruction Pipelines for Satellite Imagery},
  author={Zhang, Kai and Sun, Jin and Snavely, Noah},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}

@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
}

@data{c6tm-vw12-19,
  doi = {10.21227/c6tm-vw12},
  url = {https://dx.doi.org/10.21227/c6tm-vw12},
  author = {Le Saux, Bertrand and Yokoya, Naoto and Hänsch, Ronny and Brown, Myron},
  publisher = {IEEE Dataport},
  title = {Data Fusion Contest 2019 ({DFC2019})},
  year = {2019},
}

@article{lee2025SkyfallGS,
  title = {{Skyfall-GS}: Synthesizing Immersive {3D} Urban Scenes from Satellite Imagery},
  author = {Jie-Ying Lee and Yi-Ruei Liu and Shr-Ruei Tsai and Wei-Cheng Chang and Chung-Ho Wu and Jiewen Chan and Zhenjun Zhao and Chieh Hubert Lin and Yu-Lun Liu},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {2510.15869},
  archivePrefix = {arXiv}
}
```
