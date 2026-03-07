# =============================================================================
# Stage 1: Builder
# Compiles Ceres Solver and ColmapForVisSat from source.
# Ubuntu 18.04 is required because install_colmapforvissat.sh hard-requires
# /usr/bin/gcc-7 and /usr/bin/g++-7, which are the defaults on 18.04.
#
# Build with: docker build -t satellitesfm .
# Dependency source archives (eigen, FreeImage, glew, gflags, ceres-solver)
# are pre-placed in preprocess_sfm/ColmapForVisSat/build/__download__/ and
# COPYed into the builder — no internet access required during build.
# =============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu18.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    gcc-7 \
    g++-7 \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Clone ColmapForVisSat and pre-populate its download cache with the source
# archives that build.py would otherwise fetch from the internet.
# build.py skips downloading any file that already exists in __download__/.
RUN git clone https://github.com/Kai-46/ColmapForVisSat.git /build/ColmapForVisSat

COPY preprocess_sfm/ColmapForVisSat/build/__download__/ \
    /build/ColmapForVisSat/build/__download__/

# set -o pipefail ensures tee does not swallow build.py's exit code on failure.
RUN bash -o pipefail -c \
    "CC=/usr/bin/gcc-7 CXX=/usr/bin/g++-7 \
    python3 /build/ColmapForVisSat/scripts/python/build.py \
        --build_path /build/ColmapForVisSat/build \
        --colmap_path /build/ColmapForVisSat \
        --without_suite_sparse \
        --cuda_archs '6.0;6.1;7.0;7.5;8.0;8.6;8.9' \
    2>&1 | tee /build/ColmapForVisSat/build_log.txt"


# =============================================================================
# Stage 2: Runtime
# Copies only the ColmapForVisSat install dir (no compiler toolchain).
# Installs Miniconda + Python 3.8 environment.
# =============================================================================
FROM nvidia/cuda:11.8.0-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime shared libraries for ColmapForVisSat + build tools needed by srtm4
# (srtm4 compiles a C extension via `make` during pip install)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    build-essential \
    libboost-filesystem1.65.1 \
    libboost-program-options1.65.1 \
    libboost-graph1.65.1 \
    libboost-regex1.65.1 \
    libboost-system1.65.1 \
    libfreeimage3 \
    libgoogle-glog0v5 \
    libgflags2.2 \
    libglew2.0 \
    libgomp1 \
    libatlas3-base \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5opengl5 \
    libcgal13 \
    libcholmod3 \
    libcxsparse3 \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy ColmapForVisSat install directory to the path expected by colmap_sfm_commands.py:
#   os.path.join(os.path.dirname(__file__), 'ColmapForVisSat/build/__install__/bin/colmap')
# where __file__ is /app/preprocess_sfm/colmap_sfm_commands.py
COPY --from=builder /build/ColmapForVisSat/build/__install__ \
    /app/preprocess_sfm/ColmapForVisSat/build/__install__

# Install Miniconda 4.12.0 (py38, pre-downloaded to avoid Docker network issues;
# this version supports GLIBC 2.17+ and is compatible with Ubuntu 18.04).
# Re-download with: wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O Miniconda3-installer.sh
COPY Miniconda3-installer.sh /tmp/miniconda.sh
RUN bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# These must be set BEFORE the conda install RUN so that srtm4's C extension
# compilation can find GDAL headers/libs from the conda environment.
ENV CPATH=/usr/include/gdal:$CPATH

# Create the SatelliteSfM Python 3.8 environment.
# gdal and libtiff must be installed via conda BEFORE pip installing srtm4,
# because srtm4 builds a C extension that requires GDAL headers at compile time.
RUN conda create -y -n SatelliteSfM python=3.8 && \
    conda run -n SatelliteSfM pip install --no-cache-dir "setuptools<58" numpy && \
    conda run -n SatelliteSfM pip install --no-cache-dir --no-build-isolation "GDAL==$(gdal-config --version)" && \
    conda run -n SatelliteSfM pip install --no-cache-dir \
        numpy \
        matplotlib \
        opencv-python \
        pyexr \
        open3d \
        tqdm \
        icecream \
        imageio \
        imageio-ffmpeg \
        utm \
        pyproj \
        pymap3d \
        trimesh \
        pyquaternion \
        srtm4 \
        streamlit && \
    conda clean -afy

# Copy repository source (preprocess_sfm/ColmapForVisSat/ is excluded via .dockerignore
# so the __install__ dir placed above is not overwritten)
COPY . /app
WORKDIR /app

# Put the conda env's Python first in PATH so `python` resolves correctly
# without needing `conda activate` or `conda run`
ENV PATH=/opt/conda/envs/SatelliteSfM/bin:$PATH
ENV CONDA_DEFAULT_ENV=SatelliteSfM

CMD ["/bin/bash"]
