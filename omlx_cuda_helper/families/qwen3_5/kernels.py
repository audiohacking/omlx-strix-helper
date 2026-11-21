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

# The same recurrence with the roles turned: one lane owns one value row and
# its whole state row in registers, and the key and query of a step reach
# every lane as scalars. A step then has no shuffles and no memory in its
# path: the rows of the next step load behind the current one. q and k are
# float32 here. The sums keep the order of the kernel above, 32 sums of every
# 32nd key and a pairwise tree, so both give the same bits; this one takes
# 3.0 ms where that takes 6.8 ms for 2048 steps of 48 heads on a Radeon 8060S.
_GDN_ROWS_SOURCE = """
const uint32_t n = blockIdx.z;
const uint32_t lane = threadIdx.x;
const uint32_t dv_idx = blockIdx.y * 32 + lane;
if (n >= B * Hv || dv_idx >= Dv) return;
const uint32_t b_idx = n / Hv;
const uint32_t hv_idx = n % Hv;
const uint32_t hk_idx = hv_idx / (Hv / Hk);
constexpr int PER = Dk / 32;

// Lane L holds PER values, from L * PER on, of a step's key and query rows.
// Every lane takes each value from there as a scalar, so no load sits in the
// path of a step.
#define ROW(reg, i) \\
  __builtin_bit_cast(float, __builtin_amdgcn_readlane(reg[(i) % PER], (i) / PER))

float state[Dk];
{
  const float* in = state_in + (size_t(n) * Dv + dv_idx) * Dk;
#pragma unroll
  for (int i = 0; i < Dk; ++i) state[i] = in[i];
}
const float* q_ = q + (size_t(b_idx) * T * Hk + hk_idx) * Dk + lane * PER;
const float* k_ = k + (size_t(b_idx) * T * Hk + hk_idx) * Dk + lane * PER;
auto v_ = v + (size_t(b_idx) * T * Hv + hv_idx) * Dv + dv_idx;
auto g_ = g + size_t(b_idx) * T * Hv + hv_idx;
auto beta_ = beta + size_t(b_idx) * T * Hv + hv_idx;
auto y_ = y + (size_t(b_idx) * T * Hv + hv_idx) * Dv + dv_idx;

int kn[PER], qn[PER], kc[PER], qc[PER];
#pragma unroll
for (int j = 0; j < PER; ++j) {
  kn[j] = __builtin_bit_cast(int, k_[j]);
  qn[j] = __builtin_bit_cast(int, q_[j]);
}
float vn = gdn_f(v_[0]), gn = gdn_f(g_[0]), bn = gdn_f(beta_[0]);

for (int t = 0; t < T; ++t) {
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    kc[j] = kn[j];
    qc[j] = qn[j];
  }
  const float gate = gn, rate = bn, value = vn;
  // The next step loads behind this one.
  if (t + 1 < T) {
    q_ += Hk * Dk;
    k_ += Hk * Dk;
    v_ += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      kn[j] = __builtin_bit_cast(int, k_[j]);
      qn[j] = __builtin_bit_cast(int, q_[j]);
    }
    vn = gdn_f(v_[0]); gn = gdn_f(g_[0]); bn = gdn_f(beta_[0]);
  }
  float p[32];
#pragma unroll
  for (int i = 0; i < Dk; ++i) {
    state[i] *= gate;
    if (i < 32) p[i] = 0.0f;
    p[i % 32] += state[i] * ROW(kc, i);
  }
#pragma unroll
  for (int w = 16; w > 0; w >>= 1) {
#pragma unroll
    for (int i = 0; i < w; ++i) p[i] += p[i + w];
  }
  const float delta = (value - p[0]) * rate;
#pragma unroll
  for (int i = 0; i < Dk; ++i) {
    state[i] += delta * ROW(kc, i);
    if (i < 32) p[i] = 0.0f;
    p[i % 32] += state[i] * ROW(qc, i);
  }
#pragma unroll
  for (int w = 16; w > 0; w >>= 1) {
#pragma unroll
    for (int i = 0; i < w; ++i) p[i] += p[i + w];
  }
  gdn_store(p[0], y_);
  y_ += Hv * Dv;
}
{
  float* out = state_out + (size_t(n) * Dv + dv_idx) * Dk;
#pragma unroll
  for (int i = 0; i < Dk; ++i) out[i] = state[i];
}
#undef ROW
"""

_GDN_KERNELS: dict = {}
# (head shape, dtype) -> whether the lane-per-row kernel serves it.
_GDN_ROWS: dict = {}


def _gdn(rows: bool):
    kernel = _GDN_KERNELS.get(rows)
    if kernel is None:
        kernel = _GDN_KERNELS[rows] = mx.fast.cuda_kernel(
            name="omlx_gated_delta_rows" if rows else "omlx_gated_delta",
            input_names=["q", "k", "v", "g", "beta", "state_in"],
            output_names=["y", "state_out"],
            header=_GDN_HEADER,
            source=_GDN_ROWS_SOURCE if rows else _GDN_SOURCE,
        )
    return kernel


def _gdn_launch(rows: bool, q, k, v, g, beta, state):
    b, t, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    dtype = q.dtype
    if rows:
        q, k = q.astype(mx.float32), k.astype(mx.float32)
    # The grid is in threads. A group of the first kernel is (32, 4, 1): Dv
    # threads cover the value axis and one group per head covers the batch.
    # A group of the second is one wave of 32 value rows.
    return _gdn(rows)(
        inputs=[q, k, v, g, beta, state],
        template=[("B", b), ("T", t), ("Hk", hk), ("Hv", hv), ("Dk", dk), ("Dv", dv)],
        grid=(32, dv // 32 if rows else dv, b * hv),
        threadgroup=(32, 1 if rows else 4, 1),
        output_shapes=[(b, t, hv, dv), (b, hv, dv, dk)],
        output_dtypes=[dtype, mx.float32],
    )


def _gdn_rows(q: mx.array, v: mx.array) -> bool:
    """Whether the lane-per-row kernel equals the other one for these heads."""
    _, _, hk, dk = q.shape
    hv, dv = v.shape[2], v.shape[3]
    shape = (hk, hv, dk, dv, q.dtype, v.dtype)
    if shape not in _GDN_ROWS:
        # A lane keeps Dk state values in registers; a wave has 256.
        ok = dk % 32 == 0 and dk <= 128 and dv % 32 == 0
        if ok:
            try:
                steps = 5
                tq = (0.1 * mx.random.normal((1, steps, hk, dk))).astype(q.dtype)
                tk = (0.1 * mx.random.normal((1, steps, hk, dk))).astype(q.dtype)
                tv = mx.random.normal((1, steps, hv, dv)).astype(v.dtype)
                tg = mx.sigmoid(mx.random.normal((1, steps, hv)))
                tb = mx.sigmoid(mx.random.normal((1, steps, hv)))
                ts = 0.1 * mx.random.normal((1, hv, dv, dk))
                got = _gdn_launch(True, tq, tk, tv, tg, tb, ts)
                want = _gdn_launch(False, tq, tk, tv, tg, tb, ts)
                ok = all(mx.array_equal(a, b).item() for a, b in zip(got, want))
                if not ok:
                    logger.warning("Lane-per-row gated-delta kernel differs; not used")
            except Exception:
                logger.warning("Lane-per-row gated-delta kernel failed", exc_info=True)
                ok = False
        _GDN_ROWS[shape] = ok
    return _GDN_ROWS[shape]


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
    y, new_state = _gdn_launch(_gdn_rows(q, v), q, k, v, g, beta, state)
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
// The backend's rms_norm adds the squares of a row four per thread with a
// fused multiply-add, then across the wave in this order. One wave here owns
// one row of NORM_DIM values, so its sum has the same bits.
constexpr uint32_t NORM_DIM = 128;
__device__ inline float glue_square(float x, float sum) { return fmaf(x, x, sum); }
__device__ inline float glue_normalizer(float squares, float eps) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    squares += __shfl_xor(squares, offset);
  }
  return 1.0f / sqrtf(squares / NORM_DIM + eps);
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

# The whole step before the recurrence: the conv and SiLU above, then the
# rms_norm of each query and key head times its scale. q, k and v leave as
# three arrays, so nothing slices the conv output afterwards.
_CONV_NORM = """
const uint32_t group = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
const uint32_t lane = threadIdx.x % 32;
const uint32_t t = blockIdx.y;
const uint32_t channels = x_shape[2];
if (group * NORM_DIM >= channels) return;
float value[4];
float squares = 0.0f;
for (uint32_t j = 0; j < 4; ++j) {
  const uint32_t c = group * NORM_DIM + lane * 4 + j;
  float acc = 0.0f;
  for (uint32_t i = 0; i < TAPS; ++i) {
    const uint32_t row = t + i;
    const float tap = (row < TAPS - 1)
        ? glue_f(window[size_t(row) * channels + c])
        : glue_f(x[size_t(row - (TAPS - 1)) * channels + c]);
    const float product = tap * w[c * TAPS + i];
    acc = (i == 0) ? product : acc + product;
  }
  const float conv = glue_round(acc, q);
  const float sigmoid = glue_round(glue_sigmoid(conv), q);
  value[j] = glue_round(conv * sigmoid, q);
  squares = glue_square(value[j], squares);
}
const uint32_t heads = KEYS / NORM_DIM;
if (group >= 2 * heads) {
  auto* out = v + (size_t(t) * (channels - 2 * KEYS) +
                   (group - 2 * heads) * NORM_DIM + lane * 4);
  for (uint32_t j = 0; j < 4; ++j) glue_store(value[j], out + j);
  return;
}
const float normalizer = glue_normalizer(squares, eps[0]);
const bool key = group >= heads;
const float scale = glue_f(scales[key ? 1 : 0]);
auto* out = (key ? k : q) +
    (size_t(t) * KEYS + (group - (key ? heads : 0)) * NORM_DIM + lane * 4);
for (uint32_t j = 0; j < 4; ++j) {
  const float normalized = glue_round(value[j] * normalizer, out);
  glue_store(normalized * scale, out + j);
}
"""

# The whole step after it: rms_norm of each value head with its weight, then
# the gate below.
_NORM_GATE = """
const uint32_t row = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
const uint32_t lane = threadIdx.x % 32;
if (row >= ROWS) return;
const size_t at = size_t(row) * NORM_DIM + lane * 4;
float value[4];
float squares = 0.0f;
for (uint32_t j = 0; j < 4; ++j) {
  value[j] = glue_f(x[at + j]);
  squares = glue_square(value[j], squares);
}
const float normalizer = glue_normalizer(squares, eps[0]);
for (uint32_t j = 0; j < 4; ++j) {
  const float normalized = glue_round(value[j] * normalizer, y);
  const float weighted = glue_round(glue_f(w[lane * 4 + j]) * normalized, y);
  const float gate = glue_f(z[at + j]);
  glue_store((gate * glue_sigmoid(gate)) * weighted, y + at + j);
}
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


def _glue(
    name: str, inputs: list[str], source: str, dtype: mx.Dtype, outputs: tuple = ("y",)
):
    # A kernel object compiles for the dtypes of its first call only.
    kernel = _GLUE_KERNELS.get((name, dtype))
    if kernel is None:
        kernel = _GLUE_KERNELS[name, dtype] = mx.fast.cuda_kernel(
            name=f"omlx_{name}_{str(dtype).rsplit('.', 1)[-1]}",
            input_names=inputs,
            output_names=list(outputs),
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


# Width of the heads the two norm kernels take: one wave per head row.
NORM_DIM = 128


def conv_norm(
    window: mx.array,
    x: mx.array,
    taps: mx.array,
    keys: int,
    scales: mx.array,
    eps: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """(q, k, v) of a linear layer from its projected rows x [1, rows, C].

    q and k are the first two ``keys`` channels of ``conv_silu`` in heads of
    ``NORM_DIM``, each ``rms_norm`` times its entry of ``scales``; v is the
    rest. ``eps`` is the norm's, one float32.
    """
    rows, channels = x.shape[1], x.shape[2]
    heads = keys // NORM_DIM
    names = ["window", "x", "w", "scales", "eps"]
    kernel = _glue("conv_norm", names, _CONV_NORM, x.dtype, ("q", "k", "v"))
    return tuple(
        kernel(
            inputs=[window.astype(x.dtype), x, taps, scales.astype(x.dtype), eps],
            template=[("TAPS", taps.shape[1]), ("KEYS", keys)],
            grid=(_launch(channels // 4), rows, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[
                (1, rows, heads, NORM_DIM),
                (1, rows, heads, NORM_DIM),
                (1, rows, channels // NORM_DIM - 2 * heads, NORM_DIM),
            ],
            output_dtypes=[x.dtype] * 3,
        )
    )


def norm_gate(x: mx.array, z: mx.array, weight: mx.array, eps: mx.array) -> mx.array:
    """``silu(z) * rms_norm(x, weight)`` over heads of ``NORM_DIM``.

    The gate product is float32 and rounds once to the type of x.
    """
    rows = x.size // NORM_DIM
    flat = (rows, NORM_DIM)
    names = ["x", "z", "w", "eps"]
    return _glue("norm_gate", names, _NORM_GATE, x.dtype)(
        inputs=[x.reshape(flat), z.astype(x.dtype).reshape(flat), weight.astype(x.dtype), eps],
        template=[("ROWS", rows)],
        grid=(_THREADS, -(-rows * 32 // _THREADS), 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[flat],
        output_dtypes=[x.dtype],
    )[0].reshape(x.shape)


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
    try:
        cached = lib.omlx_gufo_attention_cached
    except AttributeError:
        logger.warning("Gufo attention library is out of date; rebuild %s", path)
        _GUFO_LIB = False
        return None
    cached.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_uint] * 4
    cached.restype = None
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
    q, out = _query_rows(queries)
    k, v = _kv_rows(keys), _kv_rows(values)
    padded = (total + 15) & ~15
    packed_k = mx.zeros((kv * padded * dim,), dtype=mx.float16)
    packed_v = mx.zeros((kv * padded * dim,), dtype=mx.float16)
    _call(
        lib.omlx_gufo_attention,
        [q, k, v, out, packed_k, packed_v],
        total - rows,
        rows,
        1 if rows >= 1024 else 0,
    )
    return out[:rows].reshape(1, rows, heads, dim).astype(queries.dtype)


def _query_rows(queries: mx.array) -> tuple[mx.array, mx.array]:
    """Queries as float32 [rows to a tile of 32, heads * dim], and the output."""
    _, heads, rows, dim = queries.shape
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
    return q, mx.zeros((tiled, heads * dim), dtype=mx.float32)


def _kv_rows(x: mx.array) -> mx.array:
    """[1, heads, n, dim] as float16 [n, heads * dim]."""
    _, heads, n, dim = x.shape
    return mx.contiguous(x.astype(mx.float16).transpose(0, 2, 1, 3).reshape(n, heads * dim))


def _call(function, arrays: list[mx.array], *numbers: int) -> None:
    """Run a library kernel on the device memory of materialized arrays."""
    mx.eval(arrays)
    held = [_device_ptr(array) for array in arrays]
    function(*(address for address, _ in held), *numbers)
    del held


# Positions a packed cache starts with, 134 MB a layer. It doubles when a
# prompt outgrows it: a 64K prompt copies it once, at 32K positions.
_PACKED_FIRST = 32768


class PackedCache:
    """Keys and values of one attention layer as the WMMA kernel reads them.

    Keys are float16 [4, capacity, 256]; values are [4, capacity / 16, 256, 16],
    each tile of 16 positions transposed. A row keeps its place as the prompt
    grows, so a chunk packs its own rows and nothing is copied per chunk.
    The kernel writes both arrays in place.
    """

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.capacity = 0
        self.length = 0

    def reserve(self, length: int) -> None:
        if length <= self.capacity:
            return
        grown = max(self.capacity, _PACKED_FIRST)
        while grown < length:
            grown *= 2
        more = grown - self.capacity
        keys = mx.zeros((4, more, 256), dtype=mx.float16)
        values = mx.zeros((4, more // 16, 256, 16), dtype=mx.float16)
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=1)
            values = mx.concatenate([self.values, values], axis=1)
        mx.eval(keys, values)
        self.keys, self.values, self.capacity = keys, values, grown

    def arrays(self) -> list[mx.array]:
        return [] if self.keys is None else [self.keys, self.values]


def gufo_cached(
    queries: mx.array, packed: PackedCache, keys: mx.array, values: mx.array, total: int
):
    """``gufo_attention`` over a packed cache, or None when the shape differs.

    ``keys`` / ``values`` [1, 4, n, 256] are the positions from
    ``packed.length`` to ``total``; they join the cache, and the last rows of
    it, one per query row, attend over all of it.
    """
    lib = _gufo_lib()
    if lib is None or queries.ndim != 4 or keys.ndim != 4:
        return None
    _, heads, rows, dim = queries.shape
    if (
        heads != 24
        or dim != 256
        or keys.shape != (1, 4, total - packed.length, 256)
        or values.shape != keys.shape
        or rows < 1
        or total < rows
        or (rows % 32 != 0 and not _GUFO_PARTIAL)
    ):
        return None
    q, out = _query_rows(queries)
    k, v = _kv_rows(keys), _kv_rows(values)
    packed.reserve(total)
    _call(
        lib.omlx_gufo_attention_cached,
        [q, k, v, out, packed.keys, packed.values],
        packed.length,
        total - rows,
        rows,
        packed.capacity,
    )
    packed.length = total
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
