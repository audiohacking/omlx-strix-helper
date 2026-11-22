# omlx strix worker

<img width="887" height="395" alt="omlxstrix" src="https://github.com/user-attachments/assets/5315dd92-b5ca-4dd1-a921-001e36028cbd" />


A prefill accelerator for AMD Strix Halo that pairs with oMLX as a CUDA helper.

oMLX can hand the front of a long prompt to a second machine while the Mac keeps decode, the KV cache, and the API. The oMLX 0.8.0 release does this with a DGX Spark or RTX A6000 GPUs. This worker puts a Strix Halo (Radeon 8060S, gfx1151) in the same seat. It speaks the same discovery, pairing, control, and data protocol, so the oMLX Cluster tab works with it as is.

The package name is `omlx_cuda_helper`, so the oMLX protocol tests import it unchanged. In an oMLX tree it lives at `apps/omlx-strixworker`.

Setup is in [DEMO.md](DEMO.md): find the device, Pair, and Check.

## Status

Alpha. Qwen3.8 27B and Qwen3.8 Flash-Next have optimized kernels on the Radeon; the other families run on portable ops.

## Results

Prefill tok/s on a 64K-token prompt with an M3 Ultra (512 GB), oQ4e checkpoints, and the increase over the M3 Ultra alone. Decode always stays on the Mac.

| Model | M3 Ultra alone | + Strix Halo | + DGX Spark | + 2x RTX A6000 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 352 | 831 (+136%) | 1,062 (+202%) | 1,879 (+434%) |
| Qwen3.8 Flash-Next | 1,370 | 2,160 (+97%\*) | 2,715 (+98%) | 3,928 (+187%) |

The M3 Ultra alone, DGX Spark, and RTX A6000 figures are from the [oMLX 0.8.0 release](https://github.com/jundot/omlx/releases/tag/v0.8.0), over 2.5 GbE. The Strix Halo figures are over Thunderbolt.

\* Against the same Mac alone, 1,095 tok/s.

M3 Ultra with one Strix Halo, 64 generated tokens, MTP on:

| Model | Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Qwen3.8 27B | 1,024 | 312 | 71 |
| Qwen3.8 27B | 8,192 | 862 | 67 |
| Qwen3.8 27B | 32,768 | 938 | 163 |
| Qwen3.8 27B | 65,536 | 831 | 66 |
| Qwen3.8 Flash-Next | 65,536 | ~2,160 | — |

## The cable

**10 Gb, no patch.** A direct 10G cable, or both machines on a 10G switch. The port reports a media speed, so stock oMLX selects it. Use this when you have it.

**Thunderbolt or USB, with the patch.** The Mac puts that port in a bridge, which has a link-local address and no media speed, so stock oMLX stays on gigabit. The [audiohacking/omlx](https://github.com/audiohacking/omlx) fork ranks the bridge by the speed the Halo reports, 80 Gbit/s. If a 10G path is also up, leave the bridge down: while the bridge answers, the fork prefers it.

After Pair, Check shows the cable in use. 10G reads **10000 Mbit/s** and the bridge reads **80000 Mbit/s**. **1000 Mbit/s** is the wrong cable.

<img width="1130" height="644" alt="image" src="https://github.com/user-attachments/assets/85a1424a-5e0e-4ff4-94d1-4e2dde91eb0e" />

## How it works

The Mac holds the weights, the cache, and generation. For prompts of 2,048 tokens and up, the Halo runs the first K layers of each prompt chunk and returns the hidden state plus the cache rows for those layers; the Mac runs the remaining layers. Both machines work at the same time on different chunks, so the Halo adds speed even when it is slower than the Mac. oMLX picks K from the measured pace of each side, or you can set it per model. Shorter prompts, and prompts with images, stay on the Mac.

On the Radeon, every kernel is checked against the plain MLX ops when a model loads, and a layer that fails a check keeps the plain ops.

**Qwen3.8 27B**

- **Gated delta**: one HIP launch per layer, with each lane holding one value row of the recurrent state in registers. `patches/rocm-cuda-kernel-hip.patch` forwards `mx.fast.cuda_kernel` to the HIP JIT.
- **Attention**: Gufo's WMMA kernel (MIT, `gufo_attention.hip`), adapted for 24 query heads, 4 KV heads, and head dimension 256. Each layer keeps its keys and values packed for the kernel, so a new chunk only adds its own rows.
- **Glue**: the causal conv with its norms, the output gate, and the MLP activation are one kernel each. Dense linears run on hipBLASLt.

**Qwen3.8 Flash-Next**

- **Experts**: gate, up, and SwiGLU in one launch and down in another, computed straight from the oQ4e codes with WMMA. The expert banks are stored in the order the kernel reads them.
- **Sparse attention**: each query row attends to its 512 selected key blocks. The block selection keeps each row's scores in registers across several waves, so its cost stays low at long contexts.
- **Hyper-connections**: fused norm, mix, and write kernels. Each dense linear is stored in whichever layout hipBLASLt multiplies faster for its shape.

## Models

The same six families and contract version as the CUDA helper.

| Family | `model_type` | On the Radeon |
| --- | --- | --- |
| Qwen3.8 27B | `qwen3_5` | Optimized kernels |
| Qwen3.8 Flash-Next | `qwen4_exp` | Optimized kernels |
| MiniMax M3 / M3.1 | `minimax_m3_vl` | Portable ops |
| GLM-5.3 Flash | `glm5_next` | Portable ops |
| DeepSeek V4.1 | `deepseek_v41` | Portable ops |
| Inkling 2 Small | `inkling2` | Portable ops |

Every family matches oMLX's recorded contract output on gfx1151. oQh checkpoints stay on the Mac, and checkpoints with 3-bit experts are not supported on the Halo yet.

## Run with Docker

The image contains the helper only; oMLX stays on the Mac. It is the Strix Halo alternative to [`jundot/omlx-cuda-helper`](https://github.com/jundot/omlx/blob/main/apps/omlx-cuda-helper/README.md), with the same discovery, pairing, ports, and data frames.

`ghcr.io/audiohacking/omlx-strix-helper` is linux/amd64 for gfx1151, built on each release of this repo. The Halo needs a ROCm-capable kernel and the render devices passed through:

```bash
docker run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
docker logs -f omlx-helper
```

Host networking lets the Cluster tab find the helper. The volume holds the identity, the pairings, the model files oMLX copies over, and the kernel cache. Arguments after the image name go to `serve`, for example `--name halo`.

Rootless podman does not see the `render` and `video` group names, so pass the caller's groups instead. The container also has to outlive the login that started it (`loginctl enable-linger`):

```bash
podman run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add keep-groups \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
```

To update, run `docker pull ghcr.io/audiohacking/omlx-strix-helper`, then `docker rm -f omlx-helper` and the same `docker run`. The volume keeps the pairing and the model files. Other commands run in the container, for example `docker exec omlx-helper omlx-strixworker --state-dir /data status`.

To build the image yourself, run `docker build -t omlx-strix-helper .` in this folder. That compiles the worker and the MLX library it imports (`mlx-build/`). It does not build oMLX.

## Build

The attention library is a gfx1151 binary and stays out of git. Build it on the Halo:

```bash
hipcc -O3 -shared -fPIC --offload-arch=gfx1151 -std=c++20 \
  omlx_cuda_helper/families/qwen3_5/gufo_attention.hip \
  -o omlx_cuda_helper/families/qwen3_5/libgufo_attn.so
```

The worker builds on MLX `fc344c9` with the mlxcel ROCm overlay and ROCm `10.3.0a20261120`. `patches/nvfp4-gfx1151-hipcc.patch` lets hipcc 7.18 compile `nvfp4_gather_expert_batched_kernel` for gfx1151.

## Tests

From an oMLX checkout:

```bash
.venv/bin/pytest apps/omlx-cuda-helper/tests \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```

Each family's test replays its oMLX contract fixture through this worker. They run on Metal on a Mac and on ROCm on the Halo; the CUDA-kernel and oQh cases skip.
