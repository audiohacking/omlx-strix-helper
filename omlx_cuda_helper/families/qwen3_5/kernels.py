# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Qwen3.5 prefill.

A block takes a kernel only after ``enable_kernels`` matched it with the
block's plain-ops reference.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def available() -> bool:
    return mx.cuda.is_available()


def attention_matches(got: mx.array, want: mx.array) -> bool:
    """Whether fused attention agrees with the plain-ops reference.

    cuDNN's fused attention moves 16-bit outputs by about 3e-3 relative to
    the float32 reference; a wrong mask or layout moves them far more.
    """
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    error = (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()
    if error <= 1e-2:
        return True
    logger.warning(
        "Fused attention differs from its reference (relative error %.2e); "
        "using the reference",
        error,
    )
    return False


_THREADS = 256

# Conversions the kernels share; the 16-bit types round to nearest even.
_HEADER = """
__device__ inline float to_float(float x) { return x; }
__device__ inline float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline float to_float(__half x) { return __half2float(x); }
__device__ inline void from_float(float x, float* out) { *out = x; }
__device__ inline void from_float(float x, __nv_bfloat16* out) {
  *out = __float2bfloat16(x);
}
__device__ inline void from_float(float x, __half* out) { *out = __float2half(x); }
"""

_CAUSAL_CONV = """
  const size_t elem = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t rows = x_shape[1] - (TAPS - 1);
  const size_t channels = x_shape[2];
  if (elem >= rows * channels) return;
  const size_t t = elem / channels;
  const size_t c = elem % channels;
  // Multiply and add separately: a fused multiply-add would round differently.
  float acc = __fmul_rn(to_float(x[t * channels + c]), w[c * TAPS]);
  for (int i = 1; i < TAPS; ++i) {
    const float tap = to_float(x[(t + i) * channels + c]);
    acc = __fadd_rn(acc, __fmul_rn(tap, w[c * TAPS + i]));
  }
  from_float(acc, y + elem);
"""
_CAUSAL_CONV_KERNELS: dict = {}


def causal_conv(x: mx.array, taps: mx.array) -> mx.array:
    """``layers.causal_conv`` in one launch, with the same float32 rounding."""
    # A kernel object compiles for the dtypes of its first call only.
    kernel = _CAUSAL_CONV_KERNELS.get(x.dtype)
    if kernel is None:
        kernel = _CAUSAL_CONV_KERNELS[x.dtype] = mx.fast.cuda_kernel(
            name=f"qwen35_causal_conv_{str(x.dtype).rsplit('.', 1)[-1]}",
            input_names=["x", "w"],
            output_names=["y"],
            source=_CAUSAL_CONV,
            header=_HEADER,
        )
    rows = x.shape[1] - taps.shape[1] + 1
    channels = x.shape[2]
    threads = -(-rows * channels // _THREADS) * _THREADS
    return kernel(
        inputs=[x, taps],
        template=[("TAPS", taps.shape[1])],
        grid=(threads, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, rows, channels)],
        output_dtypes=[x.dtype],
    )[0]


# One GPU launch for the whole sequence. The ROCm build of mlx otherwise
# expands gated_delta_update into a graph per token, and Qwen3.5 spends
# almost every prefill second there. cuda_kernel on this build is the HIP
# JIT (see patches/rocm-cuda-kernel-hip.patch). Metal and CUDA keep the
# mlx primitive, which is already fused there.
_HIP_JIT: bool | None = None


def hip_jit() -> bool:
    """True when ``mx.fast.cuda_kernel`` compiles and runs a HIP kernel."""

    global _HIP_JIT
    if _HIP_JIT is not None:
        return _HIP_JIT
    try:
        probe = mx.fast.cuda_kernel(
            name="omlx_hip_probe",
            input_names=["x"],
            output_names=["y"],
            source="y[0] = x[0];",
        )
        x = mx.array([1.0])
        y = probe(
            inputs=[x],
            output_shapes=[(1,)],
            output_dtypes=[mx.float32],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )[0]
        mx.eval(y)
        _HIP_JIT = float(y.item()) == 1.0
    except Exception:
        _HIP_JIT = False
    return _HIP_JIT


_GDN_HEADER = """
__device__ inline float gdn_f(float x) { return x; }
__device__ inline float gdn_f(__half x) { return __half2float(x); }
__device__ inline float gdn_f(hip_bfloat16 x) { return static_cast<float>(x); }
__device__ inline void gdn_store(float x, float* o) { *o = x; }
__device__ inline void gdn_store(float x, __half* o) { *o = __float2half(x); }
__device__ inline void gdn_store(float x, hip_bfloat16* o) { *o = hip_bfloat16(x); }
"""

# Port of the Metal gated-delta kernel: one wave owns 32 key lanes, four
# value lanes share the group, and the recurrence stays in registers.
# State stays float32, matching mx.fast.gated_delta_update. Gate and beta
# are [B, T, Hv]. Heads are repeated when Hv is a multiple of Hk.
_GDN_SOURCE = """
auto n = blockIdx.z;
auto dv_idx = blockIdx.y * blockDim.y + threadIdx.y;
auto dk_idx = threadIdx.x;
if (n >= B * Hv || dv_idx >= Dv) return;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;

float state[n_per_t];
auto state_in_ = state_in + (n * Dv + dv_idx) * Dk + dk_idx;
auto state_out_ = state_out + (n * Dv + dv_idx) * Dk + dk_idx;
for (int i = 0; i < n_per_t; ++i) {
  state[i] = state_in_[i * 32];
}

auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk + dk_idx;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk + dk_idx;
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv + dv_idx;
auto g_ = g + b_idx * T * Hv + hv_idx;
auto beta_ = beta + b_idx * T * Hv + hv_idx;
auto y_ = y + b_idx * T * Hv * Dv + hv_idx * Dv + dv_idx;

for (int t = 0; t < T; ++t) {
  float gate = gdn_f(g_[0]);
  float rate = gdn_f(beta_[0]);
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    state[i] *= gate;
    kv_mem += state[i] * gdn_f(k_[i * 32]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    kv_mem += __shfl_xor(kv_mem, offset);
  }
  float delta = (gdn_f(v_[0]) - kv_mem) * rate;
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    state[i] += delta * gdn_f(k_[i * 32]);
    out += state[i] * gdn_f(q_[i * 32]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    out += __shfl_xor(out, offset);
  }
  if (dk_idx == 0) {
    gdn_store(out, y_);
  }
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y_ += Hv * Dv;
  g_ += Hv;
  beta_ += Hv;
}
for (int i = 0; i < n_per_t; ++i) {
  state_out_[i * 32] = state[i];
}
"""

_gdn_kernel = None


def _gdn():
    global _gdn_kernel
    if _gdn_kernel is None:
        _gdn_kernel = mx.fast.cuda_kernel(
            name="omlx_gated_delta",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["y", "state_out"],
            header=_GDN_HEADER,
            source=_GDN_SOURCE,
        )
    return _gdn_kernel


def gated_delta(q, k, v, g, beta, state):
    """Fused gated delta, or None when this shape has to use the mlx op.

    ``state`` may be None on the first chunk. The returned state is float32
    with layout ``[B, Hv, Dv, Dk]``, the same one mlx keeps.
    """

    if g.ndim != 3 or beta.ndim != 3:
        return None
    b, t, hk, dk = q.shape
    if k.shape != q.shape or v.shape[0] != b or v.shape[1] != t:
        return None
    hv, dv = v.shape[2], v.shape[3]
    if hv % hk != 0 or dk % 32 != 0 or dv % 4 != 0 or dk == 0:
        return None
    if g.shape != (b, t, hv) or beta.shape != (b, t, hv):
        return None
    if state is None:
        state = mx.zeros((b, hv, dv, dk), dtype=mx.float32)
    elif state.shape != (b, hv, dv, dk) or state.dtype != mx.float32:
        return None
    # The grid is in threads. One group is (32, 4, 1), so Dv threads cover
    # the value axis and one group per head covers the batch.
    y, new_state = _gdn()(
        inputs=[q, k, v, g, beta, state],
        template=[("B", b), ("T", t), ("Hk", hk), ("Hv", hv), ("Dk", dk), ("Dv", dv)],
        grid=(32, dv, b * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(b, t, hv, dv), (b, hv, dv, dk)],
        output_dtypes=[q.dtype, mx.float32],
    )
    return y, new_state


# The elementwise steps between the matrix products, one launch each. As plain
# MLX ops every step is its own kernel over tens of megabytes: the causal conv
# alone took 10.6 ms of a 2048-row chunk in each linear layer, against 0.6 ms
# here. Every kernel rounds where the ops it replaces round, and a layer takes
# one only after its output equals the plain ops bit for bit.
_GLUE_HEADER = """
// Multiply and add stay separate, as the plain ops round them.
#pragma clang fp contract(off)
__device__ inline float glue_f(float x) { return x; }
__device__ inline float glue_f(__half x) { return __half2float(x); }
__device__ inline float glue_f(hip_bfloat16 x) { return static_cast<float>(x); }
__device__ inline void glue_store(float x, float* o) { *o = x; }
__device__ inline void glue_store(float x, __half* o) { *o = __float2half(x); }
__device__ inline void glue_store(float x, hip_bfloat16* o) { *o = hip_bfloat16(x); }
// x as the output type holds it.
template <typename T> __device__ inline float glue_round(float x, T*) {
  T rounded;
  glue_store(x, &rounded);
  return glue_f(rounded);
}
// The Sigmoid of this backend's unary ops.
__device__ inline float glue_sigmoid(float x) {
  const float y = 1.0f / (1.0f + expf(-fabsf(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}
"""

# silu(causal_conv([window; x], w)). The taps add up in float32 in order and
# the sum rounds once; the SiLU then rounds its sigmoid and its product.
_CONV_SILU = """
const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
const uint32_t t = blockIdx.y;
const uint32_t channels = x_shape[2];
if (c >= channels) return;
float acc = 0.0f;
for (uint32_t i = 0; i < TAPS; ++i) {
  const uint32_t row = t + i;
  const float value = (row < TAPS - 1)
      ? glue_f(window[size_t(row) * channels + c])
      : glue_f(x[size_t(row - (TAPS - 1)) * channels + c]);
  const float product = value * w[c * TAPS + i];
  acc = (i == 0) ? product : acc + product;
}
auto* out = y + (size_t(t) * channels + c);
const float conv = glue_round(acc, out);
const float sigmoid = glue_round(glue_sigmoid(conv), out);
glue_store(conv * sigmoid, out);
"""

# silu(g) * u in the type of g.
_SILU_MUL = """
const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
if (e >= uint32_t(g_shape[1])) return;
const float gate = glue_f(g[e]);
const float sigmoid = glue_round(glue_sigmoid(gate), y + e);
const float act = glue_round(gate * sigmoid, y + e);
glue_store(act * glue_f(u[e]), y + e);
"""

# silu(z) * x in float32, rounded once to the type of x.
_GATE_MUL = """
const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
if (e >= uint32_t(x_shape[1])) return;
const float gate = glue_f(z[e]);
glue_store((gate * glue_sigmoid(gate)) * glue_f(x[e]), y + e);
"""

_GLUE_KERNELS: dict = {}


def _glue(name: str, inputs: list[str], source: str, dtype: mx.Dtype):
    # A kernel object compiles for the dtypes of its first call only.
    kernel = _GLUE_KERNELS.get((name, dtype))
    if kernel is None:
        kernel = _GLUE_KERNELS[name, dtype] = mx.fast.cuda_kernel(
            name=f"omlx_{name}_{str(dtype).rsplit('.', 1)[-1]}",
            input_names=inputs,
            output_names=["y"],
            header=_GLUE_HEADER,
            source=source,
        )
    return kernel


def _launch(count: int) -> int:
    return -(-count // _THREADS) * _THREADS


def conv_silu(window: mx.array, x: mx.array, taps: mx.array) -> mx.array:
    """``silu(causal_conv([window; x], taps))`` for x [1, rows, C]."""
    rows, channels = x.shape[1], x.shape[2]
    return _glue("conv_silu", ["window", "x", "w"], _CONV_SILU, x.dtype)(
        inputs=[window.astype(x.dtype), x, taps],
        template=[("TAPS", taps.shape[1])],
        grid=(_launch(channels), rows, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def _elementwise(name: str, inputs: list[str], source: str, a: mx.array, b: mx.array):
    return _glue(name, inputs, source, a.dtype)(
        inputs=[a.reshape(1, -1), b.astype(a.dtype).reshape(1, -1)],
        grid=(_launch(a.size), 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, a.size)],
        output_dtypes=[a.dtype],
    )[0].reshape(a.shape)


def silu_mul(gate: mx.array, up: mx.array) -> mx.array:
    """``nn.silu(gate) * up`` for two arrays of one shape and type."""
    return _elementwise("silu_mul", ["g", "u"], _SILU_MUL, gate, up)


def gate_mul(x: mx.array, z: mx.array) -> mx.array:
    """``silu(z) * x`` in float32, rounded once to the type of x."""
    return _elementwise("gate_mul", ["x", "z"], _GATE_MUL, x, z)


# Gufo's gfx1151 WMMA attention for the Qwen 24/4/256 shape. MLX's own flash
# kernel is about 2 TFLOP/s here; this one holds ~30. The source and the
# shared library are built on the Strix. See gufo_attention.hip.
_GUFO_LIB = None
_GUFO_READY: bool | None = None
# The kernel masks a partial 32-row tile. Proved on this GPU before a short
# final chunk (64K ends on 2047 rows) is allowed off the slow fallback.
_GUFO_PARTIAL = False


def _gufo_lib():
    global _GUFO_LIB
    if _GUFO_LIB is False:
        return None
    if _GUFO_LIB is not None:
        return _GUFO_LIB
    import ctypes
    from pathlib import Path

    path = Path(__file__).resolve().parent / "libgufo_attn.so"
    try:
        lib = ctypes.CDLL(str(path))
    except OSError:
        logger.warning("Gufo attention library is not built (%s)", path)
        _GUFO_LIB = False
        return None
    lib.omlx_gufo_attention.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_int,
    ]
    lib.omlx_gufo_attention.restype = None
    _GUFO_LIB = lib
    return lib


def _device_ptr(array: mx.array):
    """Device pointer of a materialized array, plus the capsule that owns it."""

    import ctypes

    capsule = array.__dlpack__()
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype = ctypes.c_char_p
    api.PyCapsule_GetName.argtypes = [ctypes.py_object]
    api.PyCapsule_GetPointer.restype = ctypes.c_void_p
    api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    name = api.PyCapsule_GetName(capsule)
    managed = api.PyCapsule_GetPointer(capsule, name)

    class _Device(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class _Tensor(ctypes.Structure):
        _fields_ = [("data", ctypes.c_void_p), ("device", _Device), ("ndim", ctypes.c_int)]

    return ctypes.cast(managed, ctypes.POINTER(_Tensor)).contents.data, capsule


def gufo_attention(
    queries: mx.array, keys: mx.array, values: mx.array, *, allow_partial: bool = False
):
    """Causal attention with Gufo's WMMA kernel, or None when the shape differs.

    ``queries`` [1, 24, rows, 256], ``keys`` / ``values`` [1, 4, total, 256].
    A partial last tile of 32 query rows is masked inside the kernel. That
    path stays off until ``gufo_ready`` has matched it, so a short final
    chunk cannot silently fall through to the slow MLX kernel. Returns
    [1, rows, 24, 256] in ``queries``' dtype.
    """

    lib = _gufo_lib()
    if lib is None or queries.ndim != 4 or keys.ndim != 4:
        return None
    _, heads, rows, dim = queries.shape
    _, kv, total, vdim = keys.shape
    partial = allow_partial or _GUFO_PARTIAL
    if (
        heads != 24
        or kv != 4
        or dim != 256
        or vdim != 256
        or values.shape != keys.shape
        or rows < 1
        or total < rows
        or (rows % 32 != 0 and not partial)
    ):
        return None
    tiled = (rows + 31) & ~31
    q = mx.contiguous(
        queries.astype(mx.float32).transpose(0, 2, 1, 3).reshape(rows, heads * dim)
    )
    if tiled != rows:
        # The last tile still issues its loads. Keep that row inside the buffer.
        q = mx.contiguous(
            mx.concatenate(
                [q, mx.zeros((tiled - rows, heads * dim), dtype=mx.float32)], axis=0
            )
        )
    k = mx.contiguous(keys.astype(mx.float16).transpose(0, 2, 1, 3).reshape(total, kv * dim))
    v = mx.contiguous(values.astype(mx.float16).transpose(0, 2, 1, 3).reshape(total, kv * dim))
    out = mx.zeros((tiled, heads * dim), dtype=mx.float32)
    padded = (total + 15) & ~15
    packed_k = mx.zeros((kv * total * dim,), dtype=mx.float16)
    packed_v = mx.zeros((kv * padded * dim,), dtype=mx.float16)
    mx.eval(q, k, v, out, packed_k, packed_v)
    held = []

    def pointer(array):
        address, capsule = _device_ptr(array)
        held.append(capsule)
        return address

    lib.omlx_gufo_attention(
        pointer(q),
        pointer(k),
        pointer(v),
        pointer(out),
        pointer(packed_k),
        pointer(packed_v),
        total - rows,
        rows,
        1 if rows >= 1024 else 0,
    )
    del held
    return out[:rows].reshape(1, rows, heads, dim).astype(queries.dtype)


def _attention_case(rows: int, total: int, *, allow_partial: bool = False):
    try:
        queries = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16) * 0.2
        keys = mx.random.normal((1, 4, total, 256)).astype(mx.bfloat16) * 0.2
        values = mx.random.normal(keys.shape).astype(mx.bfloat16) * 0.2
        got = gufo_attention(queries, keys, values, allow_partial=allow_partial)
        want = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0 / 256**0.5, mask="causal"
        ).transpose(0, 2, 1, 3)
        return got is not None and attention_matches(got, want)
    except Exception:
        logger.warning(
            "Gufo attention check failed for %d rows of %d", rows, total, exc_info=True
        )
        return False


def gufo_ready() -> bool:
    """Whether the Gufo kernel matches plain attention on this GPU."""

    global _GUFO_READY, _GUFO_PARTIAL
    if _GUFO_READY is not None:
        return _GUFO_READY
    if _gufo_lib() is None:
        _GUFO_READY = False
        return False
    if not _attention_case(128, 192):
        _GUFO_READY = False
        return False
    # Unpacked partial tile, and a packed partial tile (the 64K tail is 2047).
    _GUFO_PARTIAL = _attention_case(33, 96, allow_partial=True) and _attention_case(
        1057, 2048, allow_partial=True
    )
    _GUFO_READY = True
    if _GUFO_PARTIAL:
        logger.info("Using the gfx1151 WMMA attention kernel")
    else:
        logger.info("Using the gfx1151 WMMA attention kernel for full query tiles")
    return _GUFO_READY
