# syntax=docker/dockerfile:1
# SPDX-License-Identifier: Apache-2.0
# omlx-strix-helper for linux/amd64 (Strix Halo, gfx1151). Build from this folder:
#   docker build -t omlx-strix-helper .
ARG ROCM=10.3.0a20261120
ARG ROCM_INDEX=https://nightly.repo.amd.com/rocm/whl-next/

FROM ubuntu:24.04 AS build
ARG ROCM
ARG ROCM_INDEX
ARG CMAKE_BUILD_PARALLEL_LEVEL=""
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git g++ cmake ninja-build python3 python3-dev \
        python3-venv libopenblas-dev liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/helper \
    && /opt/helper/bin/pip install --no-cache-dir -U pip setuptools wheel
ENV PATH=/opt/helper/bin:$PATH
# The gfx1151 device library is a wheel. hipcc does not need a GPU to target it.
RUN pip install --no-cache-dir --index-url "$ROCM_INDEX" \
        "rocm[libraries,devel,device-gfx1151]==${ROCM}" \
    && ln -sfn "$(rocm-sdk path --root)" /opt/rocm
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/helper/bin:/opt/rocm/bin:$PATH
COPY mlx-build /src/mlx-build
RUN git clone --quiet --filter=blob:none https://github.com/ml-explore/mlx /src/mlx \
    && /src/mlx-build/apply_patches.sh /src/mlx
RUN CMAKE_BUILD_PARALLEL_LEVEL="$CMAKE_BUILD_PARALLEL_LEVEL" \
        /src/mlx-build/build_wheel.sh /src/mlx /wheels
COPY pyproject.toml README.md /src/helper/
COPY omlx_cuda_helper /src/helper/omlx_cuda_helper
RUN hipcc -O3 -shared -fPIC --offload-arch=gfx1151 -std=c++20 \
        /src/helper/omlx_cuda_helper/families/qwen3_5/gufo_attention.hip \
        -o /src/helper/omlx_cuda_helper/families/qwen3_5/libgufo_attn.so
RUN pip install --no-cache-dir /wheels/mlx-*.whl /src/helper

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 iproute2 libopenblas0 liblapacke libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/helper /opt/helper
ENV PATH=/opt/helper/bin:$PATH
# The SDK lives inside the venv. One symlink, no second copy of it.
RUN ln -sfn "$(rocm-sdk path --root)" /opt/rocm
ENV PATH=/opt/helper/bin:/opt/rocm/bin:$PATH \
    ROCM_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib
# Identity, pairing keys, model files and the kernel cache.
VOLUME /data
EXPOSE 8850 8851 8852
ENTRYPOINT ["omlx-strixworker", "--state-dir", "/data", "serve"]
