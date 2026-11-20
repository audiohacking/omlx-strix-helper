# omlx-strixworker

Prefill helper for an AMD Strix Halo machine. oMLX drives it as the CUDA helper: same discovery, pairing, control channel, and data framing. This repo is the only place that code is committed. The oMLX checkout holds it as `apps/omlx-strixworker` while it is being tested.

The import package is `omlx_cuda_helper` on purpose. The protocol tests in the oMLX tree import that name, and this process has to answer them.

## Protocol tests

From the oMLX checkout, so this package is imported instead of `apps/omlx-cuda-helper`:

```bash
.venv/bin/pytest apps/omlx-cuda-helper/tests/test_protocol.py \
  apps/omlx-cuda-helper/tests/test_chain.py \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```

`test_qwen3_5.py` replays the oMLX contract fixture through this package's stage. On a Mac that is the Metal path, the same one the CUDA helper uses when CUDA is absent. On the Halo it is the ROCm path. The CUDA-only oQh checks stay skipped until that kernel is ported.

The Halo build is MLX `fc344c9` with the mlxcel ROCm overlay, ROCm `10.3.0a20261120`, and `patches/nvfp4-gfx1151-hipcc.patch`. hipcc 7.18 aborts while compiling `nvfp4_gather_expert_batched_kernel` for gfx1151, so that launch falls through to `nvfp4_qmv`. Qwen3.5 affine does not use it. With that patch the same contract test passed on the Radeon 8060S: 10 passed, 5 skipped.

```bash
.venv/bin/pytest apps/omlx-cuda-helper/tests/test_qwen3_5.py \
  apps/omlx-cuda-helper/tests/test_protocol.py \
  apps/omlx-cuda-helper/tests/test_chain.py \
  -o 'pythonpath=apps/omlx-strixworker apps/omlx-cuda-helper/tests' -q
```
