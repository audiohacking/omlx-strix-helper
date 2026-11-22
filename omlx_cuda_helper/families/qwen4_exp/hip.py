# SPDX-License-Identifier: Apache-2.0
"""HIP kernels of the Qwen3.8 Flash-Next blocks for gfx1151.

The CUDA kernels of this family are tensor-core code. These take their place
on a Radeon through ``mx.fast.cuda_kernel``, which this MLX build forwards to
the HIP JIT. A WMMA op multiplies 16x16 bf16 tiles into float32 sums: lane
``l`` of a wave holds row ``l % 16`` of the left tile and column ``l % 16``
of the right one, and element ``i`` of its result is row ``2 i + l / 16`` of
column ``l % 16``.
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_HEADER = r"""
// A product and a sum stay two roundings, as the ops they replace round.
#pragma clang fp contract(off)
#define FN_INF __builtin_inff()
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));
typedef v4u v4u_any __attribute__((aligned(2)));
typedef v2u v2u_any __attribute__((aligned(2)));

__device__ __forceinline__ v8f fn_wmma(v16s a, v16s b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
// 16 consecutive bf16 values as one tile operand.
__device__ __forceinline__ v16s fn_load16(const hip_bfloat16* p) {
  union { v16s f; v4u u[2]; } c;
  c.u[0] = *reinterpret_cast<const v4u_any*>(p);
  c.u[1] = *reinterpret_cast<const v4u_any*>(p + 8);
  return c.f;
}
// The bits of a float rounded to bf16, to nearest even.
__device__ __forceinline__ unsigned fn_bf16(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ float fn_float(unsigned bits) {
  return __builtin_bit_cast(float, bits << 16);
}
// A float rounded to bf16 and widened again.
__device__ __forceinline__ float fn_round(float x) { return fn_float(fn_bf16(x)); }
__device__ __forceinline__ float fn_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
// 8 consecutive bf16 values widened, and 8 floats rounded and stored.
__device__ __forceinline__ void fn_load8(const hip_bfloat16* p, float* out) {
  const v4u raw = *reinterpret_cast<const v4u_any*>(p);
#pragma unroll
  for (int d = 0; d < 4; ++d) {
    out[2 * d] = fn_float(raw[d] & 0xffffu);
    out[2 * d + 1] = fn_float(raw[d] >> 16);
  }
}
__device__ __forceinline__ void fn_store8(hip_bfloat16* p, const float* in) {
  v4u raw;
#pragma unroll
  for (int d = 0; d < 4; ++d) raw[d] = fn_bf16(in[2 * d]) | (fn_bf16(in[2 * d + 1]) << 16);
  *reinterpret_cast<v4u_any*>(p) = raw;
}
__device__ __forceinline__ int fn_wave_count(int x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}
// Bit l set where lane l holds true.
__device__ __forceinline__ unsigned fn_ballot(bool x) {
  unsigned mask = x ? 1u << threadIdx.x : 0u;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mask |= (unsigned)__shfl_xor((int)mask, o);
  return mask;
}
// The sum of a float over the lanes of the wave.
__device__ __forceinline__ float fn_wave_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o);
  return x;
}
// 16 4-bit codes of two words as a tile operand: each code as bf16, exact.
__device__ __forceinline__ v16s fn_codes(const unsigned* p) {
  union { v16s f; unsigned u[8]; } c;
#pragma unroll
  for (int word = 0; word < 2; ++word) {
    const unsigned x = p[word];
#pragma unroll
    for (int d = 0; d < 4; ++d) {
      const float low = (float)((x >> (8 * d)) & 15u);
      const float high = (float)((x >> (8 * d + 4)) & 15u);
      c.u[4 * word + d] = (__builtin_bit_cast(unsigned, low) >> 16) |
                          (__builtin_bit_cast(unsigned, high) & 0xffff0000u);
    }
  }
  return c.f;
}
__device__ __forceinline__ float fn_lane(float x, int lane) {
  return __builtin_bit_cast(
      float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, x), lane));
}
"""

_AVAILABLE: bool | None = None


def available() -> bool:
    """True when the HIP JIT of this MLX build compiles and runs a bf16 WMMA op."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            probe = mx.fast.cuda_kernel(
                name="qwen4_hip_probe",
                input_names=["x"],
                output_names=["y"],
                source="y[threadIdx.x] = x[0] + fn_wmma((v16s){}, (v16s){}, (v8f){})[0];",
                header=_HEADER,
            )
            y = probe(
                inputs=[mx.array([1.0])],
                grid=(32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(32,)],
                output_dtypes=[mx.float32],
            )[0]
            _AVAILABLE = y[0].item() == 1.0
        except Exception:  # noqa: BLE001 - no JIT, or another GPU's compiler
            _AVAILABLE = False
        if not _AVAILABLE:
            logger.info("No HIP WMMA kernels on this device; using the ops")
    return _AVAILABLE


# Attention of one (query row, KV head) over the row's QSA blocks.
#
# Four waves split the head dimension, so each keeps a short accumulator and
# the four slices of a score sum in LDS before the softmax. One wave holding
# the whole head was register-bound and ran about 3x slower on gfx1151.
# Scores are taken transposed, keys as rows and the group's query heads as
# columns. One exchange of a wave's halves then leaves a lane with the 16
# scores of its head. Each weight enters the value product as two bf16 parts,
# its rounding and the remainder. Values are read from a copy held as
# [block][dim][key of the block].
#
# q [KV * G, rows, D], k [KV, keys, D], vt [KV, blocks, D, RATIO] bf16,
# sel [rows, TOPK] int32 in key order, first int32 [1]: keys row 0 sees.
# out [rows, KV * G, D] bf16. Two waves split the head. Four waves synced
# once per 16 keys and ran slower; one wave spilled and was slower too.
_QSA = r"""
constexpr int NT = D / 16;
constexpr int NW = 2;
constexpr int SL = NT / NW;
__shared__ float ps[NW][32][8];

const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
const int sub = lane & 15, half = lane >> 4;
const int R = q_shape[1];
const int row = blockIdx.x / KV, kvh = blockIdx.x % KV;
if (row >= R) return;
const int vis = first[0] + row;
const int complete = vis / RATIO;
const int count = complete < TOPK ? complete : TOPK;
const int tail = vis - complete * RATIO;
const int nblk = count + (tail > 0 ? 1 : 0);
const int* picks = sel + size_t(row) * TOPK;
const hip_bfloat16* kh = k + size_t(kvh) * k_shape[1] * D;
const hip_bfloat16* vh = vt + size_t(kvh) * vt_shape[1] * D * RATIO;
const hip_bfloat16* qh = q + (size_t(kvh * G + (sub < G ? sub : 0)) * R + row) * D;
const int ks0 = wave * SL;

float m = -FN_INF, l = 0.0f;
v8f o[SL];
#pragma unroll
for (int n = 0; n < SL; ++n) o[n] = (v8f){};

for (int b0 = 0; b0 < nblk; b0 += 16 / RATIO) {
  const int j = b0 + sub / RATIO;
  int blk = 0;
  if (j < count) blk = picks[j];
  else if (j == count) blk = complete;
  const hip_bfloat16* kp = kh + (size_t(blk) * RATIO + sub % RATIO) * D;

  v8f s = (v8f){};
#pragma unroll
  for (int n = 0; n < SL; ++n)
    s = fn_wmma(fn_load16(kp + (ks0 + n) * 16), fn_load16(qh + (ks0 + n) * 16), s);
#pragma unroll
  for (int i = 0; i < 8; ++i) ps[wave][lane][i] = s[i];
  __syncthreads();
  s = (v8f){};
#pragma unroll
  for (int w = 0; w < NW; ++w)
#pragma unroll
    for (int i = 0; i < 8; ++i) s[i] += ps[w][lane][i];

  // Element i is key 2 i + half of this lane's head: take the other half's.
  float sc[16];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float other = __shfl_xor(s[i], 16);
    sc[2 * i] = half ? other : s[i];
    sc[2 * i + 1] = half ? s[i] : other;
  }
  float tile_max = -FN_INF;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    const int at = b0 + e / RATIO;
    const bool valid = at < count || (at == count && e % RATIO < tail);
    sc[e] = valid ? sc[e] * INV_SQRT_D : -FN_INF;
    tile_max = fmaxf(tile_max, sc[e]);
  }
  const float next = fmaxf(m, tile_max);
  const float alpha = m == -FN_INF ? 0.0f : expf(m - next);
  float sum = 0.0f;
  v16s ph, pl;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    const float p = sc[e] == -FN_INF ? 0.0f : expf(sc[e] - next);
    sum += p;
    const unsigned high = fn_bf16(p);
    ph[e] = (short)high;
    pl[e] = (short)fn_bf16(p - fn_float(high));
  }
  l = l * alpha + sum;
  m = next;

  // The sums so far are rows of heads; scale each by its head's factor.
  bool moved = false;
  float scale[16];
#pragma unroll
  for (int h = 0; h < 16; ++h) {
    scale[h] = h < G ? fn_lane(alpha, h) : 1.0f;
    moved = moved || scale[h] != 1.0f;
  }
  if (moved) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float f = half ? scale[2 * i + 1] : scale[2 * i];
#pragma unroll
      for (int n = 0; n < SL; ++n) o[n][i] *= f;
    }
  }

  int blocks[16 / RATIO];
#pragma unroll
  for (int t = 0; t < 16 / RATIO; ++t)
    blocks[t] = __builtin_amdgcn_readlane(blk, t * RATIO);
#pragma unroll
  for (int n = 0; n < SL; ++n) {
    union { v16s f; v2u u[4]; } vf;
#pragma unroll
    for (int t = 0; t < 16 / RATIO; ++t)
      vf.u[t] = *reinterpret_cast<const v2u_any*>(
          vh + (size_t(blocks[t]) * D + (ks0 + n) * 16 + sub) * RATIO);
    o[n] = fn_wmma(ph, vf.f, o[n]);
    o[n] = fn_wmma(pl, vf.f, o[n]);
  }
  __syncthreads();
}

float total[16];
#pragma unroll
for (int h = 0; h < 16; ++h) total[h] = h < G ? fn_lane(l, h) : 1.0f;
#pragma unroll
for (int i = 0; i < 8; ++i) {
  const int head = 2 * i + half;
  if (head >= G) continue;
  const float inv = half ? total[2 * i + 1] : total[2 * i];
  hip_bfloat16* y = out + (size_t(row) * KV * G + kvh * G + head) * D + sub;
#pragma unroll
  for (int n = 0; n < SL; ++n) y[(ks0 + n) * 16] = hip_bfloat16(o[n][i] / inv);
}
"""

# TR consecutive query rows attend the union of their selected blocks.
# One 16-token chunk is loaded once and scored against every row; a key a row
# did not select, or that is past that row's causal tail, scores as -inf.
# The running output stays in registers, which fits two rows and not four.
_QSA_UNION = r"""
constexpr int NT = D / 16;
constexpr int NW = 4;
constexpr int SL = NT / NW;
constexpr int TR = 2;
__shared__ float ps[NW][32][8];
__shared__ int ub[1024];
__shared__ unsigned um[1024];
__shared__ int meta[1 + 2 * TR];

const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
const int sub = lane & 15, half = lane >> 4;
const int R = q_shape[1];
const int group = blockIdx.x / KV, kvh = blockIdx.x % KV;
const int row0 = group * TR;
if (row0 >= R) return;
const hip_bfloat16* kh = k + size_t(kvh) * k_shape[1] * D;
const hip_bfloat16* vh = vt + size_t(kvh) * vt_shape[1] * D * RATIO;
const int ks0 = wave * SL;

if (threadIdx.x == 0) {
  int ptr[TR], cnt[TR], comp[TR], tail[TR];
#pragma unroll
  for (int q = 0; q < TR; ++q) {
    ptr[q] = 0;
    const int row = row0 + q;
    if (row >= R) {
      cnt[q] = 0;
      comp[q] = 0;
      tail[q] = 0;
      continue;
    }
    const int vis = first[0] + row;
    comp[q] = vis / RATIO;
    tail[q] = vis - comp[q] * RATIO;
    cnt[q] = comp[q] < TOPK ? comp[q] : TOPK;
  }
  int n = 0;
  while (n < 1024) {
    int best = 0x7fffffff;
#pragma unroll
    for (int q = 0; q < TR; ++q) {
      if (ptr[q] >= cnt[q]) continue;
      const int b = sel[(size_t(row0 + q) * TOPK) + ptr[q]];
      if (b < best) best = b;
    }
    if (best == 0x7fffffff) break;
    unsigned mask = 0;
#pragma unroll
    for (int q = 0; q < TR; ++q) {
      if (ptr[q] >= cnt[q]) continue;
      if (sel[(size_t(row0 + q) * TOPK) + ptr[q]] != best) continue;
      mask |= 1u << q;
      ptr[q]++;
    }
    ub[n] = best;
    um[n] = mask;
    n++;
  }
#pragma unroll
  for (int q = 0; q < TR; ++q) {
    if (tail[q] <= 0 || row0 + q >= R) continue;
    int found = -1;
    for (int i = 0; i < n; ++i)
      if (ub[i] == comp[q]) found = i;
    if (found < 0 && n < 1024) {
      ub[n] = comp[q];
      um[n] = 0;
      found = n++;
    }
    if (found >= 0) um[found] |= 1u << q;
  }
  meta[0] = n;
#pragma unroll
  for (int q = 0; q < TR; ++q) {
    meta[1 + q] = comp[q];
    meta[1 + TR + q] = tail[q];
  }
}
__syncthreads();
const int nblk = meta[0];

float m[TR], l[TR];
v8f o[TR][SL];
#pragma unroll
for (int q = 0; q < TR; ++q) {
  m[q] = -FN_INF;
  l[q] = 0.0f;
#pragma unroll
  for (int n = 0; n < SL; ++n) o[q][n] = (v8f){};
}

for (int b0 = 0; b0 < nblk; b0 += 16 / RATIO) {
  const int j = b0 + sub / RATIO;
  int blk = 0;
  unsigned mask = 0;
  if (j < nblk) {
    blk = ub[j];
    mask = um[j];
  }
  const hip_bfloat16* kp = kh + (size_t(blk) * RATIO + sub % RATIO) * D;
  v16s akey[SL];
#pragma unroll
  for (int n = 0; n < SL; ++n) akey[n] = fn_load16(kp + (ks0 + n) * 16);
  int blocks[16 / RATIO];
#pragma unroll
  for (int t = 0; t < 16 / RATIO; ++t)
    blocks[t] = __builtin_amdgcn_readlane(blk, t * RATIO);
  union { v16s f; v2u u[4]; } aval[SL];
#pragma unroll
  for (int n = 0; n < SL; ++n) {
#pragma unroll
    for (int t = 0; t < 16 / RATIO; ++t)
      aval[n].u[t] = *reinterpret_cast<const v2u_any*>(
          vh + (size_t(blocks[t]) * D + (ks0 + n) * 16 + sub) * RATIO);
  }

#pragma unroll 1
  for (int qi = 0; qi < TR; ++qi) {
    const int row = row0 + qi;
    const hip_bfloat16* qh =
        q + (size_t(kvh * G + (sub < G ? sub : 0)) * R + (row < R ? row : row0)) * D;
    v8f s = (v8f){};
#pragma unroll
    for (int n = 0; n < SL; ++n)
      s = fn_wmma(akey[n], fn_load16(qh + (ks0 + n) * 16), s);
#pragma unroll
    for (int i = 0; i < 8; ++i) ps[wave][lane][i] = s[i];
    __syncthreads();
    s = (v8f){};
#pragma unroll
    for (int w = 0; w < NW; ++w)
#pragma unroll
      for (int i = 0; i < 8; ++i) s[i] += ps[w][lane][i];
    float sc[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float other = __shfl_xor(s[i], 16);
      sc[2 * i] = half ? other : s[i];
      sc[2 * i + 1] = half ? s[i] : other;
    }
    float tile_max = -FN_INF;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const int at = b0 + e / RATIO;
      const unsigned bit = at < nblk ? um[at] : 0u;
      const int b = at < nblk ? ub[at] : -1;
      const bool mine = (bit >> qi) & 1u;
      const bool tail_ok = b != meta[1 + qi] || (e % RATIO) < meta[1 + TR + qi];
      const bool valid = row < R && mine && tail_ok;
      sc[e] = valid ? sc[e] * INV_SQRT_D : -FN_INF;
      tile_max = fmaxf(tile_max, sc[e]);
    }
    const float next = fmaxf(m[qi], tile_max);
    const float alpha = m[qi] == -FN_INF ? 0.0f : expf(m[qi] - next);
    float sum = 0.0f;
    v16s ph, pl;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const float p = sc[e] == -FN_INF ? 0.0f : expf(sc[e] - next);
      sum += p;
      const unsigned high = fn_bf16(p);
      ph[e] = (short)high;
      pl[e] = (short)fn_bf16(p - fn_float(high));
    }
    l[qi] = l[qi] * alpha + sum;
    m[qi] = next;
    bool moved = false;
    float scale[16];
#pragma unroll
    for (int h = 0; h < 16; ++h) {
      scale[h] = h < G ? fn_lane(alpha, h) : 1.0f;
      moved = moved || scale[h] != 1.0f;
    }
    if (moved) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float f = half ? scale[2 * i + 1] : scale[2 * i];
#pragma unroll
        for (int n = 0; n < SL; ++n) o[qi][n][i] *= f;
      }
    }
#pragma unroll
    for (int n = 0; n < SL; ++n) {
      o[qi][n] = fn_wmma(ph, aval[n].f, o[qi][n]);
      o[qi][n] = fn_wmma(pl, aval[n].f, o[qi][n]);
    }
    __syncthreads();
  }
}

#pragma unroll 1
for (int q = 0; q < TR; ++q) {
  const int row = row0 + q;
  if (row >= R) continue;
  float total[16];
#pragma unroll
  for (int h = 0; h < 16; ++h) total[h] = h < G ? fn_lane(l[q], h) : 1.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int head = 2 * i + half;
    if (head >= G) continue;
    const float inv = half ? total[2 * i + 1] : total[2 * i];
    hip_bfloat16* y = out + (size_t(row) * KV * G + kvh * G + head) * D + sub;
#pragma unroll
    for (int n = 0; n < SL; ++n) y[(ks0 + n) * 16] = hip_bfloat16(o[q][n][i] / inv);
  }
}
"""


class SparseAttention:
    """Attention of contiguous query rows over their QSA-selected keys."""

    # Query rows of one launch. An 8192-row prompt is one tile, and the rows
    # past the dense prefix stay in that launch instead of four passes.
    rows_per_launch = 8192

    @staticmethod
    def fits(heads: int, kv_heads: int, dim: int, ratio: int, dtype: mx.Dtype) -> bool:
        return (
            dtype == mx.bfloat16
            and heads % kv_heads == 0
            and heads // kv_heads <= 16
            and dim % 64 == 0
            and dim <= 256
            and ratio == 4
        )

    def __init__(self, heads: int, kv_heads: int, dim: int, ratio: int, top_k: int):
        self.heads, self.kv_heads, self.dim, self.ratio = heads, kv_heads, dim, ratio
        self.template = [
            ("KV", kv_heads),
            ("G", heads // kv_heads),
            ("D", dim),
            ("RATIO", ratio),
            ("TOPK", top_k),
        ]
        self.kernel = mx.fast.cuda_kernel(
            name=f"qwen4_hip_qsa_{heads}_{kv_heads}_{dim}_{ratio}_{top_k}",
            input_names=["q", "k", "vt", "sel", "first"],
            output_names=["out"],
            source=_QSA.replace("INV_SQRT_D", f"{float(dim) ** -0.5!r}f"),
            header=_HEADER,
            shared_memory=2 * 32 * 8 * 4 + 256,
        )

    def by_block(self, values: mx.array) -> mx.array:
        """Values [KV, keys, D] as [KV, blocks, D, ratio], zero past the keys."""
        kv, keys, dim = values.shape
        blocks = -(-keys // self.ratio)
        if blocks * self.ratio != keys:
            pad = mx.zeros((kv, blocks * self.ratio - keys, dim), dtype=values.dtype)
            values = mx.concatenate([values, pad], axis=1)
        values = values.reshape(kv, blocks, self.ratio, dim)
        return mx.contiguous(values.transpose(0, 1, 3, 2))

    def blocked(self, cache) -> mx.array:
        """The values of a QSA cache by block, extended by the rows it gained."""
        have = 0 if cache.blocked is None else cache.blocked.shape[1]
        full = self.by_block(cache.values[0, :, have * self.ratio :])
        if have:
            full = mx.concatenate([cache.blocked, full], axis=1)
        # A block that is still filling is laid out again next time.
        cache.blocked = full[:, : cache.offset // self.ratio]
        return full

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        selected: mx.array,
        first_visible: int,
        by_block: mx.array | None = None,
    ) -> mx.array:
        """``queries`` [1, heads, rows, dim] -> [1, rows, heads, dim].

        ``selected`` [rows, top_k] holds each row's block ids in key order and
        row ``i`` sees the first ``first_visible + i`` keys.
        """
        rows = queries.shape[2]
        if by_block is None:
            by_block = self.by_block(values[0])
        first = mx.array([first_visible], dtype=mx.int32)
        out = self.kernel(
            inputs=[queries[0], keys[0], by_block, selected, first],
            template=self.template,
            grid=(rows * self.kv_heads * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(rows, self.heads, self.dim)],
            output_dtypes=[queries.dtype],
        )[0]
        return out[None]

    def union(self, queries, keys, values, selected, first_visible, by_block=None):
        """The same attention, two query rows at a time over the union of their blocks."""
        kernel = getattr(self, "_union", None)
        if kernel is None:
            kernel = mx.fast.cuda_kernel(
                name=f"qwen4_hip_qsa_union_{self.heads}_{self.kv_heads}_{self.dim}",
                input_names=["q", "k", "vt", "sel", "first"],
                output_names=["out"],
                source=_QSA_UNION.replace("INV_SQRT_D", f"{float(self.dim) ** -0.5!r}f"),
                header=_HEADER,
            )
            self._union = kernel
        rows = queries.shape[2]
        if by_block is None:
            by_block = self.by_block(values[0])
        first = mx.array([first_visible], dtype=mx.int32)
        groups = -(-rows // 2)
        out = kernel(
            inputs=[queries[0], keys[0], by_block, selected, first],
            template=self.template,
            grid=(groups * self.kv_heads * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(rows, self.heads, self.dim)],
            output_dtypes=[queries.dtype],
        )[0]
        return out[None]


# Rows through 4-bit experts, a wave an (expert, block of output columns).
#
# A group of 64 inputs shares one scale and bias, so its part of a sum is
# scale * sum(x * code) + bias * sum(x). The codes are small integers, exact
# as bf16, and the first sum is a WMMA product of the rows as stored and the
# codes; the scale, the bias and the total are float32. A wave takes TM tiles
# of 16 routes at a time against TN tiles of 16 columns.
#
# The bank is held as a wave reads it (see ``expert_bank``): the codes of a
# column tile step by step, its scales and biases group by group.
#
# x [rows, K] bf16, xs [rows, K / 64] float32 sums of each group of x,
# w [E, N / 16, K / 16, 16, 2] uint32, sb [E, N / 16, K / 64, 16, 2] bf16
# (scale, bias), offsets [E + 1] int32, src [routes] uint32 (GATHER: route r
# reads row src[r]). y [routes, N] bf16,
# or with ACT [routes, N / 2] of silu(first half) * second half, each step
# rounded to bf16 as the ops round it.
# A group of 64 inputs shares one scale and bias, so its part of a sum is
# scale * sum(x * code) + bias * sum(x). The codes are small integers, exact
# as bf16, and the first sum is a WMMA product of the rows as stored and the
# codes; the scale, the bias and the total are float32. A wave takes TM tiles
# of 16 routes at a time against TN tiles of 16 columns.
#
# The bank is held as a wave reads it (see ``expert_bank``): the codes of a
# column tile step by step, its scales and biases group by group.
#
# x [rows, K] bf16, xs [rows, K / 64] float32 sums of each group of x,
# w [E, N / 16, K / 16, 16, 2] uint32, sb [E, N / 16, K / 64, 16, 2] bf16
# (scale, bias), offsets [E + 1] int32, src [routes] uint32 (GATHER: route r
# reads row src[r]). y [routes, N] bf16,
# or with ACT [routes, N / 2] of silu(first half) * second half, each step
# rounded to bf16 as the ops round it.
_EXPERTS = r"""
constexpr int GROUPS = K / 64;
constexpr int NH = ACT ? TN / 2 : TN;
constexpr int W = ACT ? N / 2 : N;
const int lane = threadIdx.x, sub = lane & 15, half = lane >> 4;
const int e = blockIdx.y;
const int begin = offsets[e], end = offsets[e + 1];
if (begin >= end) return;
const int c0 = blockIdx.x * NH * 16 + sub;
// Column tile n of the wave in the expert's tiles.
#define FN_TILE(n) \
  (size_t(e) * (N / 16) + blockIdx.x * NH + (ACT && (n) >= NH ? W / 16 + (n) - NH : (n)))

for (int r0 = begin; r0 < end; r0 += 16 * TM) {
  const int live = (end - r0 + 15) / 16;
  int rid[TM];
#pragma unroll
  for (int t = 0; t < TM; ++t) {
    const int r = r0 + t * 16 + sub;
    const int at = r < end ? r : end - 1;
    rid[t] = GATHER ? (int)src[at] : at;
  }
  int rsel[TM][8];
#pragma unroll
  for (int t = 0; t < TM; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int even = __builtin_amdgcn_readlane(rid[t], 2 * i);
      const int odd = __builtin_amdgcn_readlane(rid[t], 2 * i + 1);
      rsel[t][i] = half ? odd : even;
    }
  }
  v8f acc[TM][TN];
#pragma unroll
  for (int t = 0; t < TM; ++t) {
#pragma unroll
    for (int n = 0; n < TN; ++n) acc[t][n] = (v8f){};
  }

  for (int g = 0; g < GROUPS; ++g) {
    v8f part[TM][TN];
#pragma unroll
    for (int t = 0; t < TM; ++t) {
#pragma unroll
      for (int n = 0; n < TN; ++n) part[t][n] = (v8f){};
    }
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const int k0 = g * 64 + ks * 16;
      v16s a[TM];
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        a[t] = (v16s){};
        if (t < live) a[t] = fn_load16(x + size_t(rid[t]) * K + k0);
      }
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        const v16s codes =
            fn_codes(w + ((FN_TILE(n) * (K / 16) + k0 / 16) * 16 + sub) * 2);
#pragma unroll
        for (int t = 0; t < TM; ++t) {
          if (t < live) part[t][n] = fn_wmma(a[t], codes, part[t][n]);
        }
      }
    }
    float sums[TM][8];
#pragma unroll
    for (int t = 0; t < TM; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        sums[t][i] = t < live ? xs[size_t(rsel[t][i]) * GROUPS + g] : 0.0f;
    }
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const unsigned both = *reinterpret_cast<const unsigned*>(
          sb + ((FN_TILE(n) * GROUPS + g) * 16 + sub) * 2);
      const float scale = fn_float(both & 0xffffu), bias = fn_float(both >> 16);
#pragma unroll
      for (int t = 0; t < TM; ++t) {
        if (t < live) {
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[t][n][i] += scale * part[t][n][i] + bias * sums[t][i];
        }
      }
    }
  }

#pragma unroll
  for (int t = 0; t < TM; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = r0 + t * 16 + 2 * i + half;
      if (r >= end) continue;
      unsigned short* out = reinterpret_cast<unsigned short*>(y) + size_t(r) * W + c0;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        float v = acc[t][n][i];
        if (ACT) {
          const float gate = fn_float(fn_bf16(v));
          const float sig = fn_float(fn_bf16(1.0f / (1.0f + expf(-gate))));
          const float silu = fn_float(fn_bf16(gate * sig));
          v = silu * fn_float(fn_bf16(acc[t][n + NH][i]));
        }
        out[n * 16] = (unsigned short)fn_bf16(v);
      }
    }
  }
}
#undef FN_TILE
"""

# One wave, one 16-row tile, two output groups. The activation group sum is
# reduced in the wave while the fragments are loaded, so the host sum is gone.
# A lane holds its own row; the other half of the wave repeats it.
_EXPERTS_TIGHT_GATE = r"""
__shared__ float rowsum[16];
constexpr int GROUPS = K / 64;
constexpr int W = N / 2;
constexpr int NG = 2;
const int sub = threadIdx.x & 15, half = threadIdx.x >> 4;
const int og0 = blockIdx.y * NG;
const int e = blockIdx.z;
const int begin = offsets[e], end = offsets[e + 1];
const int r0 = begin + blockIdx.x * 16;
if (og0 >= W / 16 || r0 >= end) return;
const int row = r0 + sub;
const int live = row < end ? row : end - 1;
const int id = GATHER ? (int)src[live] : live;
v8f acc0[NG], acc1[NG];
const unsigned* wg[NG];
const unsigned* wu[NG];
#pragma unroll
for (int ogi = 0; ogi < NG; ++ogi) {
  acc0[ogi] = {};
  acc1[ogi] = {};
  const int og = og0 + ogi;
  wg[ogi] = w + ((size_t(e) * (N / 16) + og) * (K / 16)) * 32 + sub * 2;
  wu[ogi] = w + ((size_t(e) * (N / 16) + (W / 16 + og)) * (K / 16)) * 32 + sub * 2;
}
const hip_bfloat16* xp = x + (size_t)id * K;
for (int g = 0; g < GROUPS; ++g) {
  v16s a[4];
  float mine = 0.0f;
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    const hip_bfloat16* ap = xp + g * 64 + ks * 16;
    a[ks] = fn_load16(ap);
    float lo[8], hi[8];
    fn_load8(ap, lo);
    fn_load8(ap + 8, hi);
#pragma unroll
    for (int d = 0; d < 8; ++d) mine += lo[d] + hi[d];
  }
  if (half == 0) rowsum[sub] = mine;
  __syncthreads();
  float gsum[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) gsum[i] = rowsum[2 * i + half];
#pragma unroll
  for (int ogi = 0; ogi < NG; ++ogi) {
    v8f p0 = {}, p1 = {};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      p0 = fn_wmma(a[ks], fn_codes(wg[ogi] + (g * 4 + ks) * 32), p0);
      p1 = fn_wmma(a[ks], fn_codes(wu[ogi] + (g * 4 + ks) * 32), p1);
    }
    const int og = og0 + ogi;
    const unsigned bg = *reinterpret_cast<const unsigned*>(
        sb + ((size_t(e) * (N / 16) + og) * GROUPS + g) * 32 + sub * 2);
    const unsigned bu = *reinterpret_cast<const unsigned*>(
        sb + ((size_t(e) * (N / 16) + W / 16 + og) * GROUPS + g) * 32 + sub * 2);
    const float s0 = fn_float(bg & 0xffffu), b0 = fn_float(bg >> 16);
    const float s1 = fn_float(bu & 0xffffu), b1 = fn_float(bu >> 16);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc0[ogi][i] += s0 * p0[i] + b0 * gsum[i];
      acc1[ogi][i] += s1 * p1[i] + b1 * gsum[i];
    }
  }
}
#pragma unroll
for (int ogi = 0; ogi < NG; ++ogi) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = r0 + 2 * i + half;
    if (r >= end) continue;
    const float gate = fn_float(fn_bf16(acc0[ogi][i]));
    const float sig = fn_float(fn_bf16(1.0f / (1.0f + expf(-gate))));
    const float silu = fn_float(fn_bf16(gate * sig));
    const float v = silu * fn_float(fn_bf16(acc1[ogi][i]));
    reinterpret_cast<unsigned short*>(y)[(size_t)r * W + (og0 + ogi) * 16 + sub] =
        (unsigned short)fn_bf16(v);
  }
}
"""

# Down projection: one wave, four output groups, the short K streamed once.
_EXPERTS_TIGHT_DOWN = r"""
constexpr int GROUPS = K / 64;
constexpr int NG = 4;
const int sub = threadIdx.x & 15, half = threadIdx.x >> 4;
const int og0 = blockIdx.y * NG;
const int e = blockIdx.z;
const int begin = offsets[e], end = offsets[e + 1];
const int r0 = begin + blockIdx.x * 16;
if (og0 >= N / 16 || r0 >= end) return;
const int row = r0 + sub;
const int live = row < end ? row : end - 1;
const int id = GATHER ? (int)src[live] : live;
int rsel[8];
#pragma unroll
for (int i = 0; i < 8; ++i) {
  const int even = __builtin_amdgcn_readlane(id, 2 * i);
  const int odd = __builtin_amdgcn_readlane(id, 2 * i + 1);
  rsel[i] = half ? odd : even;
}
v8f acc[NG];
const unsigned* wc[NG];
#pragma unroll
for (int ogi = 0; ogi < NG; ++ogi) {
  acc[ogi] = {};
  wc[ogi] = w + ((size_t(e) * (N / 16) + og0 + ogi) * (K / 16)) * 32 + sub * 2;
}
const hip_bfloat16* xp = x + (size_t)id * K;
for (int g = 0; g < GROUPS; ++g) {
  v16s a[4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) a[ks] = fn_load16(xp + g * 64 + ks * 16);
#pragma unroll
  for (int ogi = 0; ogi < NG; ++ogi) {
    v8f p = {};
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
      p = fn_wmma(a[ks], fn_codes(wc[ogi] + (g * 4 + ks) * 32), p);
    const unsigned both = *reinterpret_cast<const unsigned*>(
        sb + ((size_t(e) * (N / 16) + og0 + ogi) * GROUPS + g) * 32 + sub * 2);
    const float scale = fn_float(both & 0xffffu), bias = fn_float(both >> 16);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[ogi][i] += scale * p[i] + bias * xs[(size_t)rsel[i] * GROUPS + g];
  }
}
#pragma unroll
for (int ogi = 0; ogi < NG; ++ogi) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int r = r0 + 2 * i + half;
    if (r >= end) continue;
    reinterpret_cast<unsigned short*>(y)[(size_t)r * N + (og0 + ogi) * 16 + sub] =
        (unsigned short)fn_bf16(acc[ogi][i]);
  }
}
"""

# Route tiles and column tiles of a wave: the fastest that keeps its sums in
# registers.
_EXPERT_TILES = (3, 2)
_EXPERT_KERNELS: dict = {}


def expert_gemm_fits(bank: tuple, width: int, act: bool = False) -> bool:
    """Whether ``expert_gemm`` covers a packed expert bank and input width."""
    weight, scales, biases, quant = bank
    if scales is None or biases is None or not quant or quant.get("oqh"):
        return False
    columns = weight.shape[1] // 2 if act else weight.shape[1]
    tiles = _EXPERT_TILES[1] // 2 if act else _EXPERT_TILES[1]
    if tiles < 1:
        return False
    return (
        quant.get("bits") == 4
        and quant.get("group_size") == 64
        and quant.get("mode", "affine") == "affine"
        and weight.dtype == mx.uint32
        and scales.dtype == mx.bfloat16
        and biases.dtype == mx.bfloat16
        and width % 64 == 0
        and weight.shape[1] % 2 == 0
        and columns % (16 * tiles) == 0
    )


def expert_bank(bank: tuple) -> tuple[mx.array, mx.array]:
    """(codes, scales and biases) of a bank in the order a wave reads them.

    Codes [E, N / 16, K / 16, 16, 2] and (scale, bias) [E, N / 16, K / 64, 16, 2]:
    what the 16 lanes of a column tile take in one step lies together.
    """
    weight, scales, biases, _ = bank
    count, width, words = weight.shape
    codes = weight.reshape(count, width // 16, 16, words // 2, 2)
    both = mx.stack([scales, biases], axis=-1)
    both = both.reshape(count, width // 16, 16, scales.shape[-1], 2)
    order = (0, 1, 3, 2, 4)
    return mx.contiguous(codes.transpose(order)), mx.contiguous(both.transpose(order))


def _tight_tiles(offsets, routes: int) -> int:
    """Row tiles of the fullest expert. Empty tiles of a shorter expert return."""
    span = mx.max(offsets[1:] - offsets[:-1])
    full = (int(span.item()) + 15) // 16
    return max(1, min(full, (routes + 15) // 16))


_TIGHT_GATE = None
_TIGHT_DOWN = None


def _expert_tight_gate(x, codes, both, offsets, src, inputs, width, count):
    global _TIGHT_GATE
    if _TIGHT_GATE is None:
        _TIGHT_GATE = mx.fast.cuda_kernel(
            name="qwen4_hip_experts_tight_gate",
            input_names=["x", "w", "sb", "offsets", "src"],
            output_names=["y"],
            source=_EXPERTS_TIGHT_GATE,
            header=_HEADER,
            shared_memory=256,
        )
    routes = x.shape[0] if src is None else src.shape[0]
    tiles = _tight_tiles(offsets, routes)
    return _TIGHT_GATE(
        inputs=[x, codes, both, offsets, offsets if src is None else src],
        template=[("K", inputs), ("N", width), ("GATHER", int(src is not None))],
        grid=(tiles * 32, (width // 2) // 32, count),
        threadgroup=(32, 1, 1),
        output_shapes=[(routes, width // 2)],
        output_dtypes=[x.dtype],
    )[0]





def _expert_tight_down(x, codes, both, offsets, src, inputs, width, count):
    global _TIGHT_DOWN
    if _TIGHT_DOWN is None:
        _TIGHT_DOWN = mx.fast.cuda_kernel(
            name="qwen4_hip_experts_tight_down",
            input_names=["x", "xs", "w", "sb", "offsets", "src"],
            output_names=["y"],
            source=_EXPERTS_TIGHT_DOWN,
            header=_HEADER,
        )
    rows = x.shape[0]
    routes = rows if src is None else src.shape[0]
    sums = x.reshape(rows, inputs // 64, 64).astype(mx.float32).sum(axis=-1)
    tiles = _tight_tiles(offsets, routes)
    return _TIGHT_DOWN(
        inputs=[x, sums, codes, both, offsets, offsets if src is None else src],
        template=[("K", inputs), ("N", width), ("GATHER", int(src is not None))],
        grid=(tiles * 32, width // 64, count),
        threadgroup=(32, 1, 1),
        output_shapes=[(routes, width)],
        output_dtypes=[x.dtype],
    )[0]


def expert_gemm(
    x: mx.array,
    bank: tuple[mx.array, mx.array],
    offsets: mx.array,
    src: mx.array | None = None,
    act: bool = False,
    tiles: tuple[int, int] | None = None,
) -> mx.array:
    """Rows of ``x`` [n, K] bf16 times each route's expert of ``bank``: [routes, N].

    ``bank`` is an ``expert_bank``. Routes are sorted by expert and
    ``offsets`` [experts + 1] bounds each expert's routes; route r reads row
    r of ``x``, or row ``src[r]`` when given. With ``act`` the bank's rows
    are the gate and then the up projection and the result is
    [routes, N / 2] of the SwiGLU.
    """
    codes, both = bank
    count, width = codes.shape[0], codes.shape[1] * 16
    rows, inputs = x.shape
    routes = rows if src is None else src.shape[0]
    tm, tn = tiles or _EXPERT_TILES
    # One wave owns two gate groups, or four down groups. The older tile loop
    # re-reads the activation once per column and loses to this on gfx1151.
    if tiles is None and inputs % 64 == 0 and act and width % 32 == 0:
        return _expert_tight_gate(
            x, codes, both, offsets, src, inputs, width, count
        )
    if tiles is None and inputs % 64 == 0 and not act and width % 64 == 0:
        return _expert_tight_down(
            x, codes, both, offsets, src, inputs, width, count
        )
    key = (tm, tn)
    kernel = _EXPERT_KERNELS.get(key)
    if kernel is None:
        kernel = _EXPERT_KERNELS[key] = mx.fast.cuda_kernel(
            name=f"qwen4_hip_experts_{tm}_{tn}",
            input_names=["x", "xs", "w", "sb", "offsets", "src"],
            output_names=["y"],
            source=_EXPERTS,
            header=_HEADER,
        )
    sums = x.reshape(rows, inputs // 64, 64).astype(mx.float32).sum(axis=-1)
    out = width // 2 if act else width
    return kernel(
        inputs=[x, sums, codes, both, offsets, offsets if src is None else src],
        template=[
            ("K", inputs),
            ("N", width),
            ("TM", tm),
            ("TN", tn),
            ("ACT", bool(act)),
            ("GATHER", src is not None),
        ],
        grid=(out // (16 * (tn // 2 if act else tn)) * 32, count, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(routes, out)],
        output_dtypes=[x.dtype],
    )[0]


_KERNELS: dict = {}


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str):
    if name not in _KERNELS:
        _KERNELS[name] = mx.fast.cuda_kernel(
            name=f"qwen4_hip_{name}",
            input_names=inputs,
            output_names=outputs,
            source=source,
            header=_HEADER,
        )
    return _KERNELS[name]


# One residual stream of a row a wave: its RMS norm as ``ops.stream_norm``
# forms it, bf16((x * inverse) * scale), and the float32 products of the
# rounded values with the inject weights.
_HC_NORM = r"""
const int lane = threadIdx.x;
const int width = x_shape[2], hidden = width / S;
const size_t row = blockIdx.x / S;
const int t = blockIdx.x % S;
const hip_bfloat16* xr = x + row * width + size_t(t) * hidden;
hip_bfloat16* yr = y + row * width + size_t(t) * hidden;
float acc = 0.0f;
for (int j = lane * 8; j < hidden; j += 256) {
  float v[8];
  fn_load8(xr + j, v);
#pragma unroll
  for (int e = 0; e < 8; ++e) acc = fmaf(v[e], v[e], acc);
}
const float inverse = 1.0f / sqrtf(fn_wave_sum(acc) / float(hidden) + eps[0]);
float dots[S];
#pragma unroll
for (int u = 0; u < S; ++u) dots[u] = 0.0f;
for (int j = lane * 8; j < hidden; j += 256) {
  float v[8];
  fn_load8(xr + j, v);
  const int at = t * hidden + j;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    v[e] = fn_round((v[e] * inverse) * scale[at + e]);
#pragma unroll
    for (int u = 0; u < S; ++u)
      dots[u] = fmaf(v[e], inject[size_t(u) * width + at + e], dots[u]);
  }
  fn_store8(yr + j, v);
}
#pragma unroll
for (int u = 0; u < S; ++u) {
  const float total = fn_wave_sum(dots[u]);
  if (lane == 0) parts[(row * S + t) * S + u] = total;
}
"""

# The block input of one hyper-connection, a thread 8 dims: sigmoid(up) in
# float32 rounded to bf16, times each stream, the streams added one by one
# and scaled by 1 / S, every step in bf16 as ``layers._stream_mixer`` does.
_HC_MIX = r"""
const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
const int width = up_shape[2], hidden = width / S;
const size_t row = i / (hidden / 8);
const int d = int(i % (hidden / 8)) * 8;
if (row >= size_t(up_shape[1])) return;
float acc[8];
#pragma unroll
for (int s = 0; s < S; ++s) {
  const size_t at = row * width + size_t(s) * hidden + d;
  float u[8], n[8];
  fn_load8(up + at, u);
  fn_load8(normed + at, n);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float term = fn_round(fn_round(fn_sigmoid(u[e])) * n[e]);
    acc[e] = s == 0 ? term : fn_round(acc[e] + term);
  }
}
#pragma unroll
for (int e = 0; e < 8; ++e) acc[e] = acc[e] * fn_round(1.0f / S);
fn_store8(out + row * hidden + d, acc);
"""

# The residual write h + branch * gate of every stream, in bf16 like
# ``layers.write``. A thread 8 dims of one row, every stream.
_HC_WRITE = r"""
const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
const int hidden = branch_shape[2];
const size_t row = i / (hidden / 8);
const int d = int(i % (hidden / 8)) * 8;
if (row >= size_t(branch_shape[1])) return;
float b[8];
fn_load8(branch + row * hidden + d, b);
#pragma unroll
for (int s = 0; s < S; ++s) {
  const float gate = fn_float(reinterpret_cast<const unsigned short*>(gates)[row * S + s]);
  const size_t at = (row * S + s) * hidden + d;
  float v[8];
  fn_load8(h + at, v);
#pragma unroll
  for (int e = 0; e < 8; ++e) v[e] = v[e] + fn_round(b[e] * gate);
  fn_store8(out + at, v);
}
"""


def hc_fits(streams: int, hidden: int, dtype: mx.Dtype) -> bool:
    return dtype == mx.bfloat16 and hidden % 256 == 0 and streams <= 8


def hc_norm(
    x: mx.array, scale: mx.array, inject: mx.array, eps: float
) -> tuple[mx.array, mx.array]:
    """(normed streams [1, L, width] bf16, float32 inject products [1, L, S]).

    ``scale`` [S, hidden] and ``inject`` [S, width] are float32.
    """
    streams, rows = scale.shape[0], x.shape[1]
    normed, parts = _kernel(
        "hc_norm", ["x", "scale", "inject", "eps"], ["y", "parts"], _HC_NORM
    )(
        inputs=[x, scale, inject, mx.array([eps], dtype=mx.float32)],
        template=[("S", streams)],
        grid=(rows * streams * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[x.shape, (1, rows, streams, streams)],
        output_dtypes=[x.dtype, mx.float32],
    )
    # Each stream's share of the products, added in stream order.
    dots = parts[:, :, 0]
    for stream in range(1, streams):
        dots = dots + parts[:, :, stream]
    return normed, dots


def hc_mix(up: mx.array, normed: mx.array, streams: int) -> mx.array:
    """The block input [1, L, hidden] from the up projection and the streams."""
    rows, width = up.shape[1], up.shape[2]
    threads = rows * (width // streams // 8)
    return _kernel("hc_mix", ["up", "normed"], ["out"], _HC_MIX)(
        inputs=[up, normed],
        template=[("S", streams)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, rows, width // streams)],
        output_dtypes=[up.dtype],
    )[0]


def hc_write(h: mx.array, branch: mx.array, gates: mx.array) -> mx.array:
    """``layers.write`` in one launch: h [1, L, S * hidden], gates [1, L, S]."""
    streams, (rows, hidden) = gates.shape[-1], branch.shape[1:]
    threads = rows * (hidden // 8)
    return _kernel("hc_write", ["h", "branch", "gates"], ["out"], _HC_WRITE)(
        inputs=[h, branch, gates],
        template=[("S", streams)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[h.shape],
        output_dtypes=[h.dtype],
    )[0]


# GDN prework of one (row, 128-channel head) a wave, 4 channels a lane: the
# causal conv in float32 rounded once, SiLU and, for the query and key heads,
# the L2 norm with ``ops.row_sum``'s bf16 lane sums, every op rounded as the
# plain ops round it.
_GDN_PREWORK = r"""
constexpr int DIM = 128;
const int lane = threadIdx.x;
const int rows = qkv_shape[1], channels = qkv_shape[2], units = channels / DIM;
const size_t row = blockIdx.x / units;
const int head = blockIdx.x % units, ch = head * DIM + lane * 4;
float acc[4];
#pragma unroll
for (int tap = 0; tap < TAPS; ++tap) {
  // Row row + tap of the conv input: the window rows, then the chunk.
  const long src = long(row) + tap - (TAPS - 1);
  const unsigned short* xp = reinterpret_cast<const unsigned short*>(
      src < 0 ? state + (src + TAPS - 1) * channels + ch : qkv + src * channels + ch);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float term = fn_float(xp[e]) * taps[(ch + e) * TAPS + tap];
    acc[e] = tap == 0 ? term : acc[e] + term;
  }
}
float act[4];
#pragma unroll
for (int e = 0; e < 4; ++e) {
  const float c = fn_round(acc[e]);
  act[e] = fn_round(c * fn_round(fn_sigmoid(c)));
}
const int section = head < KH ? 0 : (head < 2 * KH ? 1 : 2);
if (section < 2) {
  float part = fn_round(act[0] * act[0]);
#pragma unroll
  for (int e = 1; e < 4; ++e) part = fn_round(part + fn_round(act[e] * act[e]));
  const float total = fn_round(fn_wave_sum(part));
  const float inv = fn_round(1.0f / sqrtf(fn_round(total + fn_round(1e-6f))));
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    act[e] = fn_round(act[e] * inv);
    // The query also takes 1 / sqrt(128).
    if (section == 0) act[e] = fn_round(act[e] * fn_round(0.08838834764831845f));
  }
}
const int at = section == 0 ? head : (section == 1 ? head - KH : head - 2 * KH);
const int heads = section == 2 ? units - 2 * KH : KH;
unsigned short* dst = reinterpret_cast<unsigned short*>(
    section == 0 ? q : (section == 1 ? k : v)) + (row * heads + at) * DIM + lane * 4;
#pragma unroll
for (int e = 0; e < 4; ++e) dst[e] = (unsigned short)fn_bf16(act[e]);
"""

# The recurrence output's RMS norm, its weight and the output gate of one
# (row, head) a wave, as ``GatedDeltaNet.finish``: the norm rounds to bf16
# before its weight and the gate product is float32.
_GDN_FINISH = r"""
constexpr int DIM = 128;
const int lane = threadIdx.x;
const size_t at = size_t(blockIdx.x) * DIM + lane * 4;
const unsigned short* yp = reinterpret_cast<const unsigned short*>(y) + at;
const unsigned short* zp = reinterpret_cast<const unsigned short*>(z) + at;
const unsigned short* wp = reinterpret_cast<const unsigned short*>(w) + lane * 4;
unsigned short* op = reinterpret_cast<unsigned short*>(out) + at;
float f[4], ss = 0.0f;
#pragma unroll
for (int e = 0; e < 4; ++e) {
  f[e] = fn_float(yp[e]);
  ss = ss + f[e] * f[e];
}
const float inv = 1.0f / sqrtf(fn_wave_sum(ss) * (1.0f / DIM) + eps[0]);
#pragma unroll
for (int e = 0; e < 4; ++e) {
  const float t = fn_round(fn_round(f[e] * inv) * fn_float(wp[e]));
  const float zf = fn_float(zp[e]);
  const float g = SIGMOID ? fn_sigmoid(zf) : zf * fn_sigmoid(zf);
  op[e] = (unsigned short)fn_bf16(t * g);
}
"""


def gdn_glue_fits(gdn) -> bool:
    """Whether the prework and finish kernels cover a GDN block's shapes."""
    return (
        gdn.key_dim == gdn.value_dim == 128
        and gdn.qkv.weight.dtype == gdn.norm_weight.dtype == mx.bfloat16
    )


def gdn_prework(
    qkv: mx.array, state: mx.array, taps: mx.array, key_heads: int
) -> tuple[mx.array, mx.array, mx.array]:
    """(q, k, v) [1, rows, heads, 128] of ``GatedDeltaNet.prework`` in one launch.

    ``state`` [1, taps - 1, C] is the conv window before the chunk.
    """
    rows, channels = qkv.shape[1], qkv.shape[2]
    value_heads = channels // 128 - 2 * key_heads
    return _kernel("gdn_prework", ["qkv", "state", "taps"], ["q", "k", "v"], _GDN_PREWORK)(
        inputs=[qkv, state, taps],
        template=[("TAPS", taps.shape[1]), ("KH", key_heads)],
        grid=(rows * (channels // 128) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, rows, key_heads, 128),
            (1, rows, key_heads, 128),
            (1, rows, value_heads, 128),
        ],
        output_dtypes=[qkv.dtype] * 3,
    )


def gdn_finish(
    y: mx.array, z: mx.array, weight: mx.array, eps: float, sigmoid: bool
) -> mx.array:
    """``GatedDeltaNet.finish`` of y [1, rows, heads, 128], [1, rows, heads * 128]."""
    rows, heads = y.shape[1], y.shape[2]
    out = _kernel("gdn_finish", ["y", "z", "w", "eps"], ["out"], _GDN_FINISH)(
        inputs=[y, z, weight, mx.array([eps], dtype=mx.float32)],
        template=[("SIGMOID", sigmoid)],
        grid=(rows * heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[weight.dtype],
    )[0]
    return out.reshape(1, rows, -1)


# Each token's expert outputs weighted and added in the order ops.stacked_sum
# adds them, rounding to bf16 after every product and sum. A thread 8 dims.
_COMBINE = r"""
const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
const int width = y_shape[2];
const size_t token = i / (width / 8);
if (token >= size_t(w_shape[0])) return;
const int d = int(i % (width / 8)) * 8;
float part[K][8];
#pragma unroll
for (int n = 0; n < K; ++n) {
  const float weight = fn_float(reinterpret_cast<const unsigned short*>(w)[token * K + n]);
  fn_load8(y + size_t(pos[token * K + n]) * width + d, part[n]);
#pragma unroll
  for (int e = 0; e < 8; ++e) part[n][e] = fn_round(part[n][e] * weight);
}
constexpr int GROUPS = K < 8 ? K : 8;
#pragma unroll
for (int first = 0; first < GROUPS; ++first) {
#pragma unroll
  for (int n = first + 8; n < K; n += 8) {
#pragma unroll
    for (int e = 0; e < 8; ++e) part[first][e] = fn_round(part[n][e] + part[first][e]);
  }
}
#pragma unroll
for (int n = 1; n < GROUPS; ++n) {
#pragma unroll
  for (int e = 0; e < 8; ++e) part[0][e] = fn_round(part[n][e] + part[0][e]);
}
fn_store8(out + token * width + d, part[0]);
"""


def combine_fits(width: int, top_k: int, dtype: mx.Dtype) -> bool:
    return dtype == mx.bfloat16 and width % 8 == 0 and top_k <= 16


def combine_experts(y: mx.array, positions: mx.array, weights: mx.array) -> mx.array:
    """Weighted sum of each token's expert rows: [tokens, width].

    ``y`` [n, 1, width] holds one row per (token, expert) in any order,
    ``positions`` [tokens * k] the row of each token's i-th expert and
    ``weights`` [tokens, k] its weight.
    """
    tokens, k = weights.shape
    width = y.shape[-1]
    threads = tokens * (width // 8)
    return _kernel("combine", ["y", "pos", "w"], ["out"], _COMBINE)(
        inputs=[y, positions.astype(mx.uint32), weights],
        template=[("K", k)],
        grid=(-(-threads // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, width)],
        output_dtypes=[y.dtype],
    )[0]


# Float32 score of every block for every query, a wave TQ tiles of 16 / H
# queries held in registers against the block bank, 16 blocks a step: the
# products of a query head and a block are exact and add in float32, and the
# heads add up as ``Attention._block_scores`` adds them.
#
# iq [rows, H, DIM] and bank [blocks, DIM] bf16, scores [rows, blocks].
_BLOCK_SCORES = r"""
constexpr int NK = DIM / 16;
constexpr int QPT = 16 / H;
const int lane = threadIdx.x, sub = lane & 15, half = lane >> 4;
const int rows = iq_shape[0], blocks = bank_shape[0];
const int q0 = blockIdx.x * TQ * QPT;
v16s a[TQ][NK];
#pragma unroll
for (int t = 0; t < TQ; ++t) {
  const int row = q0 + t * QPT + sub / H;
  const hip_bfloat16* qp = iq + (size_t(row < rows ? row : rows - 1) * H + sub % H) * DIM;
#pragma unroll
  for (int ks = 0; ks < NK; ++ks) a[t][ks] = fn_load16(qp + ks * 16);
}
for (int b0 = 0; b0 < blocks; b0 += 16) {
  const int blk = b0 + sub < blocks ? b0 + sub : blocks - 1;
  const hip_bfloat16* bp = bank + size_t(blk) * DIM;
  v8f acc[TQ];
#pragma unroll
  for (int t = 0; t < TQ; ++t) acc[t] = (v8f){};
#pragma unroll
  for (int ks = 0; ks < NK; ++ks) {
    const v16s b = fn_load16(bp + ks * 16);
#pragma unroll
    for (int t = 0; t < TQ; ++t) acc[t] = fn_wmma(a[t][ks], b, acc[t]);
  }
#pragma unroll
  for (int t = 0; t < TQ; ++t) {
    // Element i is (query, head) 2 i + half of this lane's block.
    float sc[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float other = __shfl_xor(acc[t][i], 16);
      sc[2 * i] = half ? other : acc[t][i];
      sc[2 * i + 1] = half ? acc[t][i] : other;
    }
#pragma unroll
    for (int q = 0; q < QPT; ++q) {
      float total = fmaxf(sc[q * H], 0.0f);
#pragma unroll
      for (int h = 1; h < H; ++h) total = total + fmaxf(sc[q * H + h], 0.0f);
      const int row = q0 + t * QPT + q;
      if (half == 0 && row < rows && b0 + sub < blocks)
        scores[size_t(row) * blocks + b0 + sub] = total / INV_SCALE;
    }
  }
}
"""

# The TOPK best complete blocks of one query a wave, in block order. The
# TOPK-th largest score is found digit by digit, counting the scores at or
# above each candidate; equal scores are taken from the first block on.
#
# scores [rows, blocks] float32, not negative; first int32 [1]: tokens row 0
# sees. sel [rows, TOPK] int32.
_TOP_BLOCKS = r"""
const int lane = threadIdx.x;
const int row = blockIdx.x;
const int complete = (first[0] + row) / RATIO;
int* out = sel + size_t(row) * TOPK;
if (complete <= TOPK) {
  for (int j = lane; j < TOPK; j += 32) out[j] = j;
  return;
}
const unsigned* sp =
    reinterpret_cast<const unsigned*>(scores) + size_t(row) * scores_shape[1];
// A score as an ordered key above zero; zero past the complete blocks.
#define FN_KEY(i) ((i) < complete ? (sp[i] & 0x7fffffffu) + 1u : 0u)
unsigned edge = 0;
for (int shift = 30; shift >= 0; shift -= 3) {
  unsigned at[7];
  int count[7];
#pragma unroll
  for (int d = 0; d < 7; ++d) {
    // The top digit has two bits; a key never reaches the last value.
    at[d] = shift == 30 && d >= 3 ? 0xffffffffu : edge + (unsigned(d + 1) << shift);
    count[d] = 0;
  }
  for (int i = lane; i < complete; i += 32) {
    const unsigned key = FN_KEY(i);
#pragma unroll
    for (int d = 0; d < 7; ++d) count[d] += key >= at[d];
  }
  int digit = 0;
#pragma unroll
  for (int d = 0; d < 7; ++d) digit += fn_wave_count(count[d]) >= TOPK;
  edge += unsigned(digit) << shift;
}
int above = 0;
for (int i = lane; i < complete; i += 32) above += FN_KEY(i) > edge;
const int ties = TOPK - fn_wave_count(above);
const unsigned below = (1u << lane) - 1u;
int taken = 0, tied = 0;
for (int i = lane; i - lane < complete; i += 32) {
  const unsigned key = FN_KEY(i);
  const unsigned equal = fn_ballot(key == edge);
  const bool take =
      key > edge || (key == edge && tied + __builtin_popcount(equal & below) < ties);
  const unsigned takes = fn_ballot(take);
  if (take) out[taken + __builtin_popcount(takes & below)] = i;
  taken += __builtin_popcount(takes);
  tied += __builtin_popcount(equal);
}
#undef FN_KEY
"""


class BlockSelection:
    """Each query's ``top_k`` best blocks, in block order."""

    # Query tiles of a wave of the score kernel.
    tiles = 2

    @staticmethod
    def fits(heads: int, dim: int, dtype: mx.Dtype) -> bool:
        return (
            dtype == mx.bfloat16 and 16 % heads == 0 and dim % 16 == 0 and dim <= 128
        )

    def __init__(self, heads: int, dim: int, ratio: int, top_k: int):
        self.heads, self.dim, self.ratio, self.top_k = heads, dim, ratio, top_k
        self.scores = mx.fast.cuda_kernel(
            name=f"qwen4_hip_block_scores_{heads}_{dim}",
            input_names=["iq", "bank"],
            output_names=["scores"],
            source=_BLOCK_SCORES.replace("INV_SCALE", f"{float(dim) ** 0.5!r}f"),
            header=_HEADER,
        )
        self.top = mx.fast.cuda_kernel(
            name=f"qwen4_hip_top_blocks_{ratio}_{top_k}",
            input_names=["scores", "first"],
            output_names=["sel"],
            source=_TOP_BLOCKS,
            header=_HEADER,
        )

    def block_scores(self, index_queries: mx.array, pooled: mx.array) -> mx.array:
        """Float32 score [rows, blocks] of each block for each query."""
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        per_wave = self.tiles * (16 // self.heads)
        return self.scores(
            inputs=[index_queries, pooled[0]],
            template=[("H", self.heads), ("DIM", self.dim), ("TQ", self.tiles)],
            grid=(-(-rows // per_wave) * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, blocks)],
            output_dtypes=[mx.float32],
        )[0]

    def __call__(
        self, index_queries: mx.array, pooled: mx.array, first_visible: int
    ) -> mx.array:
        """[rows, top_k] int32 block ids; row ``i`` sees ``first_visible + i`` tokens.

        A row with at most ``top_k`` complete blocks gets the first ``top_k`` ids.
        """
        rows = index_queries.shape[0]
        return self.top(
            inputs=[
                self.block_scores(index_queries, pooled),
                mx.array([first_visible], dtype=mx.int32),
            ],
            template=[("RATIO", self.ratio), ("TOPK", self.top_k)],
            grid=(rows * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(rows, self.top_k)],
            output_dtypes=[mx.int32],
        )[0]


# ``Attention.prepare`` of one (row, head) a wave: the RMS norm with its
# float32 scale rounded once, then the rotation of the first 2 * HALF dims by
# the row's position, each product and sum rounded as the plain ops round
# them. Heads in order: queries, keys, values (copied), index queries, the
# raw index key (copied). A lane holds width / 32 consecutive dims.
_QSA_PREP = r"""
constexpr int UNITS = H + 2 * KV + IH + 1;
const int lane = threadIdx.x;
const int rows = qo_shape[1];
const size_t row = blockIdx.x / UNITS;
const int u = blockIdx.x % UNITS;
const hip_bfloat16* from;
const float* w = nullptr;
hip_bfloat16* to;
int width = D;
if (u < H) {
  from = qo + (row * H + u) * 2 * D;
  w = qs;
  to = q + (size_t(u) * rows + row) * D;
} else if (u < H + KV) {
  from = ko + (row * KV + (u - H)) * D;
  w = ks;
  to = k + (size_t(u - H) * rows + row) * D;
} else if (u < H + 2 * KV) {
  from = vo + (row * KV + (u - H - KV)) * D;
  to = v + (size_t(u - H - KV) * rows + row) * D;
} else {
  const int i = u - H - 2 * KV;
  width = ID;
  from = io + (row * (IH + 1) + i) * ID;
  w = i < IH ? iqs : nullptr;
  to = i < IH ? iq + (row * IH + i) * ID : ik + row * ID;
}
const int per = width / 32, base = lane * per;
const unsigned short* src = reinterpret_cast<const unsigned short*>(from);
unsigned short* dst = reinterpret_cast<unsigned short*>(to);
float x[8];
#pragma unroll
for (int e = 0; e < 8; ++e) x[e] = e < per ? fn_float(src[base + e]) : 0.0f;
if (w != nullptr) {
  float ss = 0.0f;
#pragma unroll
  for (int e = 0; e < 8; ++e) ss = fmaf(x[e], x[e], ss);
  const float inv = 1.0f / sqrtf(fn_wave_sum(ss) / float(width) + eps[0]);
#pragma unroll
  for (int e = 0; e < 8; ++e)
    if (e < per) x[e] = fn_round((x[e] * inv) * w[base + e]);
  // The partner of dim d < HALF is d + HALF, HALF / per lanes away.
  const int hop = HALF / per;
  const float pos = float(pos0[0] + int(row));
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float other = __shfl_xor(x[e], hop);
    const int d = base + e;
    if (e < per && d < 2 * HALF) {
      const float angle = pos * freq[d < HALF ? d : d - HALF];
      const float c = cosf(angle), sn = sinf(angle);
      x[e] = d < HALF ? x[e] * c - other * sn : x[e] * c + other * sn;
    }
  }
}
#pragma unroll
for (int e = 0; e < 8; ++e)
  if (e < per) dst[base + e] = (unsigned short)fn_bf16(x[e]);
"""


def qsa_prep_fits(heads, kv_heads, dim, index_heads, index_dim, half, dtype) -> bool:
    """Whether ``qsa_prep`` covers a QSA geometry: whole lanes and rotations."""
    return (
        dtype == mx.bfloat16
        and dim in (128, 256)
        and index_dim in (128, 256)
        and half > 0
        and half % (dim // 32) == 0
        and half % (index_dim // 32) == 0
        and 2 * half <= min(dim, index_dim)
        and all(
            (half // (w // 32)) & (half // (w // 32) - 1) == 0 for w in (dim, index_dim)
        )
    )


def qsa_prep(att, q_out, k_out, v_out, index, first: int) -> tuple[mx.array, ...]:
    """``att.prepare`` outputs but the gate: (queries, keys, values, raw index
    keys, index queries) for rows at positions ``first`` on."""
    rows, dim = q_out.shape[1], att.head_dim
    heads, kv, ih, idim = att.heads, att.kv_heads, att.index_heads, att.index_dim
    q, k, v, iq, ik = _kernel(
        "qsa_prep",
        ["qo", "ko", "vo", "io", "qs", "ks", "iqs", "freq", "pos0", "eps"],
        ["q", "k", "v", "iq", "ik"],
        _QSA_PREP,
    )(
        inputs=[
            q_out,
            k_out,
            v_out,
            index,
            att.q_scale,
            att.k_scale,
            att.index_q_scale,
            att.inv_freq,
            mx.array([first], dtype=mx.int32),
            mx.array([att.eps], dtype=mx.float32),
        ],
        template=[
            ("H", heads),
            ("KV", kv),
            ("D", dim),
            ("IH", ih),
            ("ID", idim),
            ("HALF", att.inv_freq.shape[0]),
        ],
        grid=(rows * (heads + 2 * kv + ih + 1) * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (1, heads, rows, dim),
            (1, kv, rows, dim),
            (1, kv, rows, dim),
            (rows, ih, idim),
            (1, rows, idim),
        ],
        output_dtypes=[q_out.dtype] * 5,
    )
    return q, k, v, ik, iq
