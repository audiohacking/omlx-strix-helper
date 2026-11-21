# omlx strix worker

<img width="887" height="395" alt="omlxstrix" src="https://github.com/user-attachments/assets/5315dd92-b5ca-4dd1-a921-001e36028cbd" />


A prefill accelerator for AMD Strix Halo that pairs into oMLX as the CUDA helper.

oMLX 0.8.0 hands the front of a long prompt to a second GPU and keeps decode, the KV cache, and the API on the Mac. In the 0.8.0 release that GPU is a DGX Spark or two RTX A6000s. This worker puts a Strix Halo in the same seat: Radeon 8060S, gfx1151, the same discovery, pairing, control channel, and data frames. The Cluster tab learns nothing new.

The package name is `omlx_cuda_helper`, so the oMLX protocol tests import it unchanged. In an oMLX tree it lives at `apps/omlx-strixworker`.

M3 Ultra, one Strix Halo, Thunderbolt. Find devices, Pair, and Check. The cable has to be the bridge; the setup is in [DEMO.md](DEMO.md).

<img width="1130" height="644" alt="image" src="https://github.com/user-attachments/assets/85a1424a-5e0e-4ff4-94d1-4e2dde91eb0e" />

## Results (WIP)

Qwen3.8 27B oQ4e. M3 Ultra 512 GB and one Strix Halo over Thunderbolt. 128 generated tokens, MTP on.

| Prompt | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| 1,024 | 312 | 79 |
| 8,192 | 780 | 81 |
| 32,768 | 811 | 165 |
| 65,536 | 710 | 67 |

Same model, 64K prompt, M3 Ultra 512 GB. The Spark and A6000 rows are the oMLX 0.8.0 release, a CUDA helper over 2.5 GbE. Decode stays on the Mac in every row.

| 64K prefill | tok/s |
| --- | ---: |
| M3 Ultra alone | 352 |
| + Strix Halo | 710 |
| + DGX Spark | 1,062 |
| + 2× RTX A6000 | 1,879 |

At 32,768 tokens the draft copies 94 of 98 eligible prompt tokens, and decode reads 165 tok/s. At 65,536 the Mac drafts new tokens at 67 tok/s.

## The split

The Mac holds the weights, the cache, and generation. From 2,048 tokens up, the Halo takes layers `[0, K)` and returns the hidden state plus the cache rows for that prefix. The Mac runs the layers that remain. Auto sets K from the measured pace of each side. Shorter prompts, and any prompt with an image, stay on the Mac.

## On the Radeon

Two kernels carry that prefix.

**Gated delta** is one HIP launch per layer. `patches/rocm-cuda-kernel-hip.patch` forwards `mx.fast.cuda_kernel` to the HIP JIT, and the recurrence runs on the GPU. The stock ROCm op builds a graph per token and leaves the shader arrays idle.

**Attention** is Gufo's WMMA kernel, adapted for Qwen's 24 query heads, 4 KV heads, and head dimension 256 (MIT, `gufo_attention.hip`). On gfx1151 it holds about 30 TF. The stock flash kernel holds about 2. A partial query tile is legal, so the short tail of a 64K prompt stays on this kernel.

The supported family is `qwen3_5`.

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
.venv/bin/pytest apps/omlx-cuda-helper/tests/test_qwen3_5.py \
  apps/omlx-cuda-helper/tests/test_protocol.py \
  apps/omlx-cuda-helper/tests/test_chain.py \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```

`test_qwen3_5.py` replays the oMLX contract fixture through this stage. Metal on a Mac, ROCm on the Halo. The oQh cases skip.
