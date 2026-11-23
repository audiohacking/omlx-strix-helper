# oMLX Strix/AMD Helper

<img width="887" height="395" alt="omlxstrix" src="https://github.com/user-attachments/assets/5315dd92-b5ca-4dd1-a921-001e36028cbd" />


A prefill accelerator for AMD Strix Halo that pairs with oMLX as a CUDA helper.

oMLX can hand the front of a long prompt to a second machine while the Mac keeps decode, the KV cache, and the API. The oMLX 0.8.0 release does this with a DGX Spark or RTX A6000 GPUs. This helper puts a Strix Halo (Radeon 8060S, gfx1151) in the same seat. It speaks the same discovery, pairing, control, and data protocol, so the oMLX Cluster tab works with it as is.

Status: alpha. Qwen3.8 27B, Qwen3.8 Flash-Next and Inkling 2 have optimized kernels on the Radeon; the other families run on portable ops.

## Results

Prefill tok/s on a 64K-token prompt with an M3 Ultra (512 GB), oQ4e checkpoints. Decode always stays on the Mac.

| Model | M3 Ultra alone | + Strix Halo | + DGX Spark | + 2x RTX A6000 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 27B | 404 | 831 | 1,062* | 1,879* |
| Qwen3.8 Flash-Next | 1,376 | 2,187 | 2,715* | 3,928* |
| Inkling 2 Mini | 1,398 | 2,468 | — | — |

* The DGX Spark and RTX A6000 figures are from the [oMLX 0.8.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.8.0)
* The Strix Halo figures are over Thunderbolt

M3 Ultra with one Strix Halo, 64 generated tokens, MTP on:

| Model | Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Qwen3.8 27B | 1,024 | 312 | 71 |
| Qwen3.8 27B | 8,192 | 862 | 67 |
| Qwen3.8 27B | 32,768 | 938 | 163 |
| Qwen3.8 27B | 65,536 | 831 | 66 |
| Qwen3.8 Flash-Next | 65,536 | 2,187 | — |

Inkling 2 Mini, 512 generated tokens:

| Prompt | M3 Ultra alone | + Strix Halo | Time to first token | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1,744 | 2,325 | 4.7 s → 3.5 s | 105 |
| 32,768 | 1,598 | 2,654 | 20.5 s → 12.3 s | 84 |
| 65,536 | 1,398 | 2,468 | 46.9 s → 26.6 s | 72 |

## Quick start

You need a Mac running oMLX and a Strix Halo on the same network, joined by a fast cable (see [The cable](#the-cable)). Hardware notes are in [DEMO.md](DEMO.md).

**1. On the Halo, start the helper.** The image is `ghcr.io/audiohacking/omlx-strix-helper` (linux/amd64, gfx1151). It contains everything the helper needs: the ROCm runtime, the MLX build, and the compiled kernels.

```bash
docker run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
docker logs -f omlx-helper
```

With rootless podman, pass the caller's groups instead, and let the container outlive the login that started it (`loginctl enable-linger`):

```bash
podman run -d --name omlx-helper --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri \
  --group-add keep-groups \
  --network host --ipc host \
  -v omlx-helper:/data \
  ghcr.io/audiohacking/omlx-strix-helper
```

Host networking lets the Cluster tab find the helper. The volume holds its identity, its pairings, the model files oMLX copies over, and the kernel cache. Arguments after the image name go to `serve`, for example `--name halo` for the name shown in the Cluster tab.

**2. On the Mac, run oMLX.** Use the [audiohacking/omlx](https://github.com/audiohacking/omlx) fork when the cable is Thunderbolt, and for Inkling 2 Mini; stock oMLX works over 10G for the other models.

```bash
git clone https://github.com/audiohacking/omlx.git
cd omlx
make install
.venv/bin/omlx serve --host 0.0.0.0 --port 8000
```

**3. Pair.** In oMLX, open Cluster, Find devices, Pair the Halo, then Check. Check shows the cable in use: **80000 Mbit/s** on Thunderbolt, **10000 Mbit/s** on 10G.

<img width="1130" height="644" alt="image" src="https://github.com/user-attachments/assets/85a1424a-5e0e-4ff4-94d1-4e2dde91eb0e" />

**4. Turn the helper on for a model** in its settings (Prefill helper: the paired Halo). Both machines use the same checkpoint; oMLX copies the layers the Halo needs.

## Recommended settings

`Qwen3.8-27B-oQ4e`:

- Prefill helper: the paired Halo, layer count on Auto.
- MTP on, VLM MTP off.
- Native bf16 KV. Float16 activations, TurboQuant, affine KV, and INT8 activation prefill off.
- ANE prefill on: sequence 2048, fraction 0.53, dual ANE, GDN fraction 0.5, GDN layers capped at 48, tail padding from 512 tokens. ANE and the helper work independently.
- Speculative prefill and DFlash off.

`Qwen3.8-Flash-Next-oQ4e-mtp`:

- Prefill helper: the paired Halo, layer count **20 or 21** of 48. Auto works too and lands close to that.
- MTP on. Other settings at their defaults.

`Inkling-2-Mini-oQ4e-mtp` (needs the fork):

- Prefill helper: the paired Halo, layer count **24** of 48. Auto works too and lands a little lower.
- MTP on. Other settings at their defaults.

## Models

The same families and contract versions as the CUDA helper.

| Family | `model_type` | On the Radeon |
| --- | --- | --- |
| Qwen3.8 27B | `qwen3_5` | Optimized kernels |
| Qwen3.8 Flash-Next | `qwen4_exp` | Optimized kernels |
| MiniMax M3 / M3.1 | `minimax_m3_vl` | Portable ops |
| GLM-5.3 Flash | `glm5_next` | Portable ops |
| DeepSeek V4.1 | `deepseek_v41` | Portable ops |
| Inkling 2 Small | `inkling2` | Optimized kernels |
| Inkling 2 Mini | `inkling2_loop` | Optimized kernels |

Every family matches oMLX's recorded contract output on gfx1151. oQh checkpoints stay on the Mac, and checkpoints with 3-bit experts are not supported on the Halo yet. Inkling 2 Mini runs its middle layers three times per token; splitting it inside those repeats needs the [audiohacking/omlx](https://github.com/audiohacking/omlx) fork.

## The cable

**10 Gb, no patch.** A direct 10G cable, or both machines on a 10G switch. The port reports a media speed, so stock oMLX selects it. Use this when you have it.

**Thunderbolt or USB, with the patch.** The Mac puts that port in a bridge, which has a link-local address and no media speed, so stock oMLX stays on gigabit. The [audiohacking/omlx](https://github.com/audiohacking/omlx) fork ranks the bridge by the speed the Halo reports, 80 Gbit/s. If a 10G path is also up, leave the bridge down: while the bridge answers, the fork prefers it.

**1000 Mbit/s** in Check is the wrong cable. [DEMO.md](DEMO.md) has the diagram and the details.

## Updating

```bash
docker pull ghcr.io/audiohacking/omlx-strix-helper
docker rm -f omlx-helper
```

Then the same `docker run` as above. The volume keeps the pairing, the model files, and the kernel cache. Other commands run in the container, for example `docker exec omlx-helper omlx-strixworker --state-dir /data status`. oMLX expects helper 0.1.7 or newer.

## Troubleshooting

- **Prefill close to the Mac's own speed.** Check that the cable reads 10000 or 80000 Mbit/s in Check. A gigabit path, or a helper that dropped out, looks like this.
- **First prompt after an update is slower.** With the layer count on Auto, oMLX first re-measures both sides; later prompts settle.
- **Speed drops over long sessions.** Both machines lower their GPU clocks under sustained load. After ten minutes or more of continuous long prompts, the Mac's own prefill can run about a fifth slower until it has cooled for a few minutes; keep the Halo's BIOS power mode on Balanced. Compare speeds with both machines cool.
- **Per-chunk timing.** Run the helper with `-e STRIX_CHUNK_LOG=1` to log, for each chunk, how long the Halo waited for the Mac and how long it computed. A Halo that never waits is the slower side; a larger layer count then makes things slower, a smaller one faster.

---

## How it works

The Mac holds the weights, the cache, and generation. For prompts of 2,048 tokens and up, the Halo runs the first K layers (layer visits, for a model that repeats layers) of each prompt chunk and returns the hidden state plus the cache rows for those layers; the Mac runs the remaining layers. Both machines work at the same time on different chunks, so the Halo adds speed even when it is slower than the Mac. oMLX picks K from the measured pace of each side, or you can set it per model. Shorter prompts, and prompts with images, stay on the Mac.

On the Radeon, every kernel is checked against the plain MLX ops when a model loads, and a layer that fails a check keeps the plain ops.

**Qwen3.8 27B**

- **Gated delta**: one HIP launch per layer, with each lane holding one value row of the recurrent state in registers. `patches/rocm-cuda-kernel-hip.patch` forwards `mx.fast.cuda_kernel` to the HIP JIT.
- **Attention**: Gufo's WMMA kernel (MIT, `gufo_attention.hip`), adapted for 24 query heads, 4 KV heads, and head dimension 256. Each layer keeps its keys and values packed for the kernel, so a new chunk only adds its own rows.
- **Glue**: the causal conv with its norms, the output gate, and the MLP activation are one kernel each. Dense linears run on hipBLASLt.

**Qwen3.8 Flash-Next**

- **Experts**: gate, up, and SwiGLU in one launch and down in another, computed straight from the oQ4e codes with WMMA. The expert banks are stored in the order the kernel reads them.
- **Sparse attention**: each query row attends to its 512 selected key blocks. The block selection keeps each row's scores in registers across several waves, so its cost stays low at long contexts.
- **Hyper-connections**: fused norm, mix, and write kernels. Each dense linear is stored in whichever layout hipBLASLt multiplies faster for its shape.

**Inkling 2 Mini**

- **Looped layers**: the Halo takes layer visits, not layers. Its share can reach into the second and third pass over the middle layers, so it holds each of those layers once and rebuilds the message bank each pass reads from the pass before.
- **Experts**: NVFP4 codes decode straight to bf16 inside a WMMA kernel, with the activation applied as the up projection stores. One launch sorts a chunk's rows by expert for both projections.
- **Attention**: one WMMA kernel per layer covers every query head that shares a key/value head, with the relative-position bias and the attention sink built in.
- **Glue**: the depth mix with its norms, the short convolutions with their residual adds, the clips, the output gate and the loop controllers are one kernel each.

## Build from source

**The image.** `docker build -t omlx-strix-helper .` in this folder compiles the worker, the attention library, and the MLX library it imports (`mlx-build/`). It does not build oMLX. Releases of this repo publish the image to `ghcr.io/audiohacking/omlx-strix-helper`.

**Without a container.** The worker builds on MLX `fc344c9` with the mlxcel ROCm overlay and the patch series in `mlx-build/` (`patches/rocm-cuda-kernel-hip.patch` forwards `mx.fast.cuda_kernel` to the HIP JIT; `patches/nvfp4-gfx1151-hipcc.patch` lets hipcc 7.18 compile `nvfp4_gather_expert_batched_kernel` for gfx1151), on ROCm `10.3.0a20261120`. A source install also compiles the attention library, which is a gfx1151 binary and stays out of git (the image already has it):

```bash
hipcc -O3 -shared -fPIC --offload-arch=gfx1151 -std=c++20 \
  omlx_cuda_helper/families/qwen3_5/gufo_attention.hip \
  -o omlx_cuda_helper/families/qwen3_5/libgufo_attn.so
```

Then install and serve, with `rocm-sdk` on `PATH` and `ROCM_PATH` set to that SDK:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
export PATH="$HOME/rocm-env/bin:$PATH"
export ROCM_PATH="$(rocm-sdk path --root)"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$HOME/rocm-env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
omlx-strixworker serve --name halo
```

State lives in `~/.omlx-cuda-helper`. Pass `--pair` only to join an oMLX that has not paired this helper yet. The package name is `omlx_cuda_helper`, so the oMLX protocol tests import it unchanged; in an oMLX tree it lives at `apps/omlx-strixworker`.

## Tests

From an oMLX checkout:

```bash
.venv/bin/pytest apps/omlx-cuda-helper/tests \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```

Each family's test replays its oMLX contract fixture through this worker. They run on Metal on a Mac and on ROCm on the Halo; the CUDA-kernel and oQh cases skip.
