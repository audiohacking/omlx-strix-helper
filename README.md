# omlx strix worker

<img width="887" height="395" alt="omlxstrix" src="https://github.com/user-attachments/assets/5315dd92-b5ca-4dd1-a921-001e36028cbd" />


A prefill accelerator for AMD Strix Halo that pairs into oMLX as the CUDA helper.

oMLX 0.8.0 hands the front of a long prompt to a second GPU and keeps decode, the KV cache, and the API on the Mac. In the 0.8.0 release that GPU is a DGX Spark or two RTX A6000s. This worker puts a Strix Halo in the same seat: Radeon 8060S, gfx1151, the same discovery, pairing, control channel, and data frames. The Cluster tab learns nothing new.

The package name is `omlx_cuda_helper`, so the oMLX protocol tests import it unchanged. In an oMLX tree it lives at `apps/omlx-strixworker`.

Find devices, Pair, and Check. The setup is in [DEMO.md](DEMO.md).

## The cable

**10Gb, no patch.** A direct 10G cable, or both machines on a 10G switch. The port reports a media speed, so stock oMLX selects it. Use this when you have it. It can be faster than the bridge below. The results table was not measured on 10G.

**Thunderbolt or USB, with the patch.** The Mac puts that port in a bridge. The bridge has a link-local address and no media speed, so stock oMLX stays on gigabit. The [audiohacking/omlx](https://github.com/audiohacking/omlx) fork ranks the bridge by the speed the Halo reports, 80 Gbit/s. If a 10G path is also up, leave the bridge down. While the bridge answers, the fork prefers it.

After Pair, Check shows the cable in use. 10G reads **10000 Mbit/s**. The bridge reads **80000 Mbit/s**. **1000 Mbit/s** is the wrong cable.

<img width="1130" height="644" alt="image" src="https://github.com/user-attachments/assets/85a1424a-5e0e-4ff4-94d1-4e2dde91eb0e" />

## Status
* Super Alpha!
* Paired prefill measured on `Qwen3.8-27B-oQ4e` and `Qwen3.8-Flash-Next-oQ4e-mtp`.

## Results (WIP)

Qwen3.8 27B oQ4e. M3 Ultra 512 GB and one Strix Halo over Thunderbolt. 64 generated tokens, MTP on.

| Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| 1,024 | 312 | 71 |
| 8,192 | 862 | 67 |
| 32,768 | 938 | 163 |
| 65,536 | 831 | 66 |

Prefill tok/s on a 64K prompt, oQ4e, M3 Ultra 512 GB. The Spark and A6000 columns are the [oMLX 0.8.0](https://github.com/jundot/omlx/releases/tag/v0.8.0) release, a CUDA helper over 2.5 GbE. Decode stays on the Mac in every paired row. The Strix column is this worker.

| Model | M3 Ultra alone | + Strix Halo | + DGX Spark | + 2x RTX A6000 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 352 | 831 (+136%) | 1,062 (+202%) | 1,879 (+434%) |
| Qwen3.8 Flash-Next | 1,370 | ~2,130 (+94%) | 2,715 (+98%) | 3,928 (+187%) |

The 27B Strix cell is the 65,536 row above, over Thunderbolt. The Flash-Next Strix cell is a 65,536-token prompt over Thunderbolt with the Halo on 20 of 48 layers: 2,080 to 2,154 tok/s over three runs. Auto settles at 22 and reads 1,941 to 2,096. The percentage is against this Mac alone on the same day, 1,072 to 1,118 tok/s; the 1,370 in that column is the release’s machine. The Halo runs the first layers of one chunk while the Mac runs the rest of the one before, so a helper slower than the Mac still adds its pace. oMLX cuts Flash-Next prompts into 4,096-token chunks.

At 32,768 tokens the draft copies 49 of 49 eligible prompt tokens, and decode reads 163 tok/s. At 65,536 the Mac drafts new tokens at 66 tok/s.

## The split

The Mac holds the weights, the cache, and generation. From 2,048 tokens up, the Halo takes layers `[0, K)` and returns the hidden state plus the cache rows for that prefix. The Mac runs the layers that remain. Auto sets K from the measured pace of each side. Shorter prompts, and any prompt with an image, stay on the Mac.

## On the Radeon

Three things carry that prefix.

**Gated delta** is one HIP launch per layer. `patches/rocm-cuda-kernel-hip.patch` forwards `mx.fast.cuda_kernel` to the HIP JIT, and the recurrence runs on the GPU. One lane owns one value row and keeps its state in registers, so a step waits on no memory. The stock ROCm op builds a graph per token and leaves the shader arrays idle.

**Attention** is Gufo's WMMA kernel, adapted for Qwen's 24 query heads, 4 KV heads, and head dimension 256 (MIT, `gufo_attention.hip`). On gfx1151 it holds about 30 TF. The stock flash kernel holds about 2. Each layer keeps its keys and values packed for the kernel, so a chunk adds its own rows and nothing re-copies the prompt. The call waits with its thread awake: with every core asleep the same kernel runs a quarter slower. A partial query tile is legal, so the short tail of a 64K prompt stays on this kernel.

**The steps between** the matrix products are one kernel each: the causal conv with its norms, the output gate, the MLP activation. As separate MLX ops they took a quarter of a chunk. Dense weights are stored as `[in, out]`, the operand hipBLASLt multiplies fastest.

Each kernel is checked against the plain ops when a model loads. A layer that fails a check keeps the plain ops.

## Models

The six families of the CUDA helper, at the same contract version.

| Family | `model_type` | On the Radeon |
| --- | --- | --- |
| Qwen3.8 27B | `qwen3_5` | Measured. HIP kernels. |
| Qwen3.8 Flash-Next | `qwen4_exp` | Paired prefill measured. HIP kernels. |
| MiniMax M3 / M3.1 | `minimax_m3_vl` | Fixture. Plain ops. |
| GLM-5.3 Flash | `glm5_next` | Fixture. Plain ops. |
| DeepSeek V4.1 | `deepseek_v41` | Fixture. Plain ops. |
| Inkling 2 Small | `inkling2` | Fixture. Plain ops. |

Fixture: the stage replays oMLX's contract fixture for that family on gfx1151 and matches the Mac's recorded output. No full checkpoint of MiniMax, GLM-5.3, DeepSeek V4.1, or Inkling has run here yet.

The CUDA helper's kernels for those families are tensor-core code and stay off. oQh checkpoints stay on the Mac. This MLX build has no 3-bit expert product, so a checkpoint with 3-bit experts does not run on the Halo yet.

## Run with Docker

This image is the helper, not oMLX. oMLX stays on the Mac. The image is the Strix Halo alternative to [`jundot/omlx-cuda-helper`](https://github.com/jundot/omlx/blob/main/apps/omlx-cuda-helper/README.md): same discovery, pairing, ports, and data frames, and it does not contain the oMLX server.

`ghcr.io/audiohacking/omlx-strix-helper` is linux/amd64 for gfx1151. A release of **this** repo builds and pushes it. The Halo needs a ROCm-capable kernel and the render devices passed through:

```bash
docker run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
docker logs -f omlx-helper
```

Host networking is what lets the Cluster tab find the helper. The volume holds the identity, the pairings, the model files oMLX copies over, and the kernel cache. Arguments after the image name go to `serve`, for example `--name halo`.

Rootless podman does not see the `render` and `video` group names, so pass the caller's groups instead. The container also has to outlive the login that started it (`loginctl enable-linger`):

```bash
podman run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add keep-groups \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
```

To update, run `docker pull ghcr.io/audiohacking/omlx-strix-helper`, then `docker rm -f omlx-helper` and the same `docker run`. The volume keeps the pairing and the model files. Other commands run in the container, for example `docker exec omlx-helper omlx-strixworker --state-dir /data status`. The first push is private until the package is made public in the repository settings.

To build the image yourself, run `docker build -t omlx-strix-helper .` in this folder. That compiles the worker and the MLX library it imports (`mlx-build/`, the ROCm counterpart of the CUDA helper's `mlx-build/`). It does not clone or build oMLX.

## Build

The attention library is a gfx1151 binary and stays out of git. Build it on the Halo:

```bash
hipcc -O3 -shared -fPIC --offload-arch=gfx1151 -std=c++20 \
  omlx_cuda_helper/families/qwen3_5/gufo_attention.hip \
  -o omlx_cuda_helper/families/qwen3_5/libgufo_attn.so
```

The measured build is MLX `fc344c9` with the mlxcel ROCm overlay and ROCm `10.3.0a20261120`. hipcc 7.18 stops on `nvfp4_gather_expert_batched_kernel` for gfx1151. `patches/nvfp4-gfx1151-hipcc.patch` sends that launch to `nvfp4_qmv`. Qwen3.5 affine never takes it.

## Protocol tests

From an oMLX checkout:

```bash
.venv/bin/pytest apps/omlx-cuda-helper/tests \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```

Each family's test replays its oMLX contract fixture through this worker's stage. Metal on a Mac, ROCm on the Halo. The CUDA kernel and oQh cases skip.
