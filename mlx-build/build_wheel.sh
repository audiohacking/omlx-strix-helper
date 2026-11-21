#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the patched MLX ROCm wheel from MLX_SRC into OUT_DIR.
# Needs hipcc from the ROCm SDK (see the Dockerfile). No GPU is required:
# MLX_ROCM_ARCHITECTURES selects the code object.
set -euo pipefail

MLX_SRC="${1:?usage: build_wheel.sh MLX_SRC OUT_DIR}"
OUT_DIR="${2:?usage: build_wheel.sh MLX_SRC OUT_DIR}"
ARCHS="${MLX_ROCM_ARCHITECTURES:-gfx1151}"

command -v rocm-sdk >/dev/null || { echo "error: rocm-sdk not on PATH" >&2; exit 1; }
export ROCM_PATH="${ROCM_PATH:-$(rocm-sdk path --root)}"
export PATH="$ROCM_PATH/bin:$PATH"
command -v hipcc >/dev/null || { echo "error: hipcc not on PATH ($ROCM_PATH/bin)" >&2; exit 1; }

mkdir -p "$OUT_DIR"
export CMAKE_PREFIX_PATH="$ROCM_PATH${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
# The image installs ninja, not make. CMake's default generator is Make.
export CMAKE_GENERATOR=Ninja
# One token per argument: setup.py splits CMAKE_ARGS on spaces.
export CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_ROCM_ARCHITECTURES=${ARCHS} \
-DMLX_BUILD_CUDA=OFF -DMLX_BUILD_METAL=OFF -DMLX_BUILD_GGUF=OFF \
-DCMAKE_PREFIX_PATH=$ROCM_PATH -DLAPACK_INCLUDE_DIRS=/usr/include \
-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON"

python -m pip wheel "$MLX_SRC" --no-deps -w "$OUT_DIR"
ls -l "$OUT_DIR"/mlx-*.whl
