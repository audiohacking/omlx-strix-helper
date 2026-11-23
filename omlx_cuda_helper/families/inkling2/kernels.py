# SPDX-License-Identifier: Apache-2.0
"""CUDA kernels for Inkling 2 prefill.

The attention, depth-mix and router kernels port the host's Metal kernels in
``omlx/patches/inkling2/kernels.py``: the same staging precision, softmax
steps and selection rules, on the tensor cores. Routed experts run on the
expert GEMMs the MiniMax family already ships. A kernel is used only after
``enable`` matched it with the plain-ops reference.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx

from ..deepseek_v41 import kernels as deepseek
from ..minimax_m3_vl import kernels as minimax
from . import layers

logger = logging.getLogger(__name__)

# Most depth sources the mix kernel takes: eight completed blocks and the
# partial sum (48 layers in blocks of 6).
MIX_SOURCES = 9
_FORMATS = ("nvfp4", "mxfp8")
# Affine code widths the expert tile takes (64 weights a group): what oQ
# writes for Inkling 2 experts below 4 bits, and requantized MXFP8 layers.
_AFFINE_BITS = (2, 3, 4, 8)
_on: set[str] = set()


def _rocm() -> bool:
    """Whether this build runs ``mx.fast.cuda_kernel`` on the HIP JIT."""
    if mx.cuda.is_available():
        return False
    from ..qwen3_5 import kernels as shared

    return shared.hip_jit()


def available() -> bool:
    return mx.cuda.is_available() or _rocm()


# The CUDA names these kernels use, on HIP. bf16 rounds to nearest even both
# ways; gfx1151 waves are 32 lanes, as CUDA warps are.
_HIP_COMPAT = """
#define __nv_bfloat16 hip_bfloat16
__device__ __forceinline__ float __bfloat162float(hip_bfloat16 x) { return static_cast<float>(x); }
__device__ __forceinline__ hip_bfloat16 __float2bfloat16_rn(float x) { return hip_bfloat16(x); }
#define __shfl_xor_sync(mask, value, offset) __shfl_xor(value, offset)
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
"""
# The float32 Tanh and Sigmoid of MLX's ROCm unary ops, which the JIT cannot
# include: the activation matches the plain ops only with these.
_HIP_UNARY = """
namespace mlx::core::cu {
struct Tanh {
  __device__ float operator()(float x) { return ::tanhf(x); }
};
struct Sigmoid {
  __device__ float operator()(float x) {
    const float y = 1.0f / (1.0f + expf(-fabsf(x)));
    return x < 0.0f ? 1.0f - y : y;
  }
};
}  // namespace mlx::core::cu
"""
# Kernels from other families' CUDA modules: they stay on plain ops on ROCm.
# Attention has its own RDNA 3 kernel there.
_CUDA_ONLY = ("router_logits", "experts")


def _cuda_kernel(**kwargs):
    """``mx.fast.cuda_kernel`` with the CUDA headers mapped for HIP on ROCm."""
    if _rocm():
        header = kwargs.get("header", "")
        header = "\n".join(
            _HIP_UNARY if "device/unary_ops.cuh" in line else line
            for line in header.splitlines()
            if not line.startswith("#include <cuda")
        )
        kwargs["header"] = _HIP_COMPAT + header + "\n"
    return mx.fast.cuda_kernel(**kwargs)


def enabled(name: str) -> bool:
    return name in _on


# One block runs 64 queries of one head in four warps (16 rows each) over
# 64-key tiles staged in half, as the host stages BF16 q/k/v in half. Scores
# are float32 in the log2 domain; the online softmax steps over 16 keys at a
# time, as the host's kernel does, and starts from the row's sink logit.
_ATTENTION_HEADER = """
#include <cuda_bf16.h>
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t ink_half2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void ink_mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ink_ldsm_t2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
"""
_ATTENTION = """
constexpr int BQ = 64, BK = 64, LD = D + 8;
constexpr float LOG2E = 1.4426950408889634f;
__shared__ __align__(16) __half ks[BK * LD];
__shared__ __align__(16) __half vs[BK * LD];
const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const int h = blockIdx.y, hk = h / (HQ / HKV);
const int L = dims[0], S = dims[1];
const int q0 = (blockIdx.x) * BQ;
const int r0 = q0 + warp * 16 + (lane >> 2), r1 = r0 + 8;
const int a0 = S - L + r0, a1 = S - L + r1;
const float qscale = scale[0] * LOG2E;

uint32_t qf[D / 16][4];
#pragma unroll
for (int kb = 0; kb < D / 16; ++kb) {
  const int c = kb * 16 + (lane & 3) * 2;
  float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (r0 < L) {
    const __nv_bfloat16* p = q + ((size_t)r0 * HQ + h) * D;
    v[0] = __bfloat162float(p[c]); v[1] = __bfloat162float(p[c + 1]);
    v[4] = __bfloat162float(p[c + 8]); v[5] = __bfloat162float(p[c + 9]);
  }
  if (r1 < L) {
    const __nv_bfloat16* p = q + ((size_t)r1 * HQ + h) * D;
    v[2] = __bfloat162float(p[c]); v[3] = __bfloat162float(p[c + 1]);
    v[6] = __bfloat162float(p[c + 8]); v[7] = __bfloat162float(p[c + 9]);
  }
  qf[kb][0] = ink_half2(v[0], v[1]);
  qf[kb][1] = ink_half2(v[2], v[3]);
  qf[kb][2] = ink_half2(v[4], v[5]);
  qf[kb][3] = ink_half2(v[6], v[7]);
}
float o[D / 8][4];
#pragma unroll
for (int n = 0; n < D / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
float m0 = r0 < L ? sink[(size_t)r0 * HQ + h] * LOG2E : 0.f;
float m1 = r1 < L ? sink[(size_t)r1 * HQ + h] * LOG2E : 0.f;
float l0 = 1.f, l1 = 1.f;
const __nv_bfloat16* rel0 = rel + ((size_t)(r0 < L ? r0 : 0) * HQ + h) * R;
const __nv_bfloat16* rel1 = rel + ((size_t)(r1 < L ? r1 : 0) * HQ + h) * R;

const int a_first = S - L + q0;
const int a_last = S - L + min(q0 + BQ, L) - 1;
int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
const int k_begin = (lo / 16) * 16;
// The warp's own rows bound which 16-key groups can be visible to it.
const int w_first = S - L + q0 + warp * 16;
const int w_last = min(S - L + q0 + warp * 16 + 15, S - 1);
const bool warp_on = q0 + warp * 16 < L;
const __nv_bfloat16* kbase = k + (size_t)hk * S * D;
const __nv_bfloat16* vbase = v + (size_t)hk * S * D;

for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < (BK * D / 8) / 128; ++i) {
    const int c = tid + i * 128;
    const int row = c / (D / 8), col = (c % (D / 8)) * 8;
    const int key = k0 + row;
    uint4 kw = make_uint4(0, 0, 0, 0), vw = make_uint4(0, 0, 0, 0);
    if (key < S) {
      kw = *reinterpret_cast<const uint4*>(kbase + (size_t)key * D + col);
      vw = *reinterpret_cast<const uint4*>(vbase + (size_t)key * D + col);
    }
    const __nv_bfloat16* kb16 = reinterpret_cast<const __nv_bfloat16*>(&kw);
    const __nv_bfloat16* vb16 = reinterpret_cast<const __nv_bfloat16*>(&vw);
    uint4 kh, vh;
    uint32_t* khp = reinterpret_cast<uint32_t*>(&kh);
    uint32_t* vhp = reinterpret_cast<uint32_t*>(&vh);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      khp[j] = ink_half2(__bfloat162float(kb16[2 * j]), __bfloat162float(kb16[2 * j + 1]));
      vhp[j] = ink_half2(__bfloat162float(vb16[2 * j]), __bfloat162float(vb16[2 * j + 1]));
    }
    *reinterpret_cast<uint4*>(ks + row * LD + col) = kh;
    *reinterpret_cast<uint4*>(vs + row * LD + col) = vh;
  }
  __syncthreads();
  if (!warp_on) continue;
  #pragma unroll
  for (int g = 0; g < BK / 16; ++g) {
    const int g0 = k0 + g * 16;
    // A group past every row of the warp, or before every row's window,
    // leaves the softmax as it is.
    if (g0 > w_last) break;
    if (WINDOW > 0 && g0 + 15 < w_first - WINDOW + 1) continue;
    float s[2][4];
    #pragma unroll
    for (int t = 0; t < 2; ++t) {
      s[t][0] = s[t][1] = s[t][2] = s[t][3] = 0.f;
      const __half* kr = ks + (g * 16 + t * 8 + (lane >> 2)) * LD + (lane & 3) * 2;
      #pragma unroll
      for (int kb = 0; kb < D / 16; ++kb) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr + kb * 16);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + kb * 16 + 8);
        ink_mma(s[t], qf[kb], b0, b1);
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int t = 0; t < 2; ++t)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = g0 + t * 8 + (lane & 3) * 2 + (e & 1);
        const int row = e < 2 ? r0 : r1;
        const int d = (e < 2 ? a0 : a1) - key;
        const bool vis = row < L && key < S && d >= 0 && (WINDOW == 0 || d < WINDOW);
        float x = s[t][e] * qscale;
        if (vis) {
          if (d < R) x += __bfloat162float((e < 2 ? rel0 : rel1)[d]) * LOG2E;
        } else {
          x = -INFINITY;
        }
        s[t][e] = x;
        if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float f0 = exp2f(m0 - n0), f1 = exp2f(m1 - n1);
    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int t = 0; t < 2; ++t)
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float p = exp2f(s[t][e] - (e < 2 ? n0 : n1));
        s[t][e] = p;
        if (e < 2) sum0 += p; else sum1 += p;
      }
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffff, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffff, sum1, 2);
    l0 = l0 * f0 + sum0;
    l1 = l1 * f1 + sum1;
    m0 = n0;
    m1 = n1;
    uint32_t pa[4];
    pa[0] = ink_half2(s[0][0], s[0][1]);
    pa[1] = ink_half2(s[0][2], s[0][3]);
    pa[2] = ink_half2(s[1][0], s[1][1]);
    pa[3] = ink_half2(s[1][2], s[1][3]);
    const int vrow = g * 16 + (lane & 15);
    #pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      o[n][0] *= f0; o[n][1] *= f0; o[n][2] *= f1; o[n][3] *= f1;
      uint32_t b0, b1;
      const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(vs + vrow * LD + n * 8));
      ink_ldsm_t2(addr, b0, b1);
      ink_mma(o[n], pa, b0, b1);
    }
  }
}
const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
for (int n = 0; n < D / 8; ++n) {
  const int c = n * 8 + (lane & 3) * 2;
  if (r0 < L) {
    __nv_bfloat16* p = out + ((size_t)r0 * HQ + h) * D + c;
    p[0] = __float2bfloat16_rn(o[n][0] * i0);
    p[1] = __float2bfloat16_rn(o[n][1] * i0);
  }
  if (r1 < L) {
    __nv_bfloat16* p = out + ((size_t)r1 * HQ + h) * D + c;
    p[0] = __float2bfloat16_rn(o[n][2] * i1);
    p[1] = __float2bfloat16_rn(o[n][3] * i1);
  }
}
"""


# The same attention on RDNA 3 WMMA (wave32, float32 out), for ROCm. A block
# runs BQ queries of every query head on one KV head, a wave per 16 queries
# of a head, over BK-key tiles staged in LDS: keys row by row in bf16, values
# transposed in fp16. Scores are taken
# transposed, keys as rows and queries as columns, so a lane owns one query
# and 8 of the 16 keys of a group; the other 8 sit in the lane 16 apart. The
# value sums are taken transposed too (dims as rows), so a lane's
# accumulators all belong to its own query. Softmax, sink, bias band and
# fp16 staging follow the CUDA kernel above.
_ATTENTION_HIP_HEADER = """
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
typedef _Float16 ink_v16h __attribute__((ext_vector_type(16)));
typedef short ink_v16s __attribute__((ext_vector_type(16)));
typedef float ink_v8f __attribute__((ext_vector_type(8)));
typedef unsigned int ink_v4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ ink_v8f ink_wmma(ink_v16h a, ink_v16h b, ink_v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ ink_v8f ink_wmma_bf(ink_v16s a, ink_v16s b, ink_v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
__device__ __forceinline__ float ink_bf(unsigned short bits) {
  return __builtin_bit_cast(float, unsigned(bits) << 16);
}
// The value of lane i ^ 16: a VALU swap of the wave's halves, no LDS trip.
__device__ __forceinline__ float ink_other_half(float x) {
  const unsigned u = __builtin_bit_cast(unsigned, x);
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlanex16(u, u, 0x76543210u, 0xfedcba98u, false, false));
}
__device__ __forceinline__ ink_v16s ink_row16(const unsigned short* p) {
  union { ink_v16s v; ink_v4u u[2]; } c;
  c.u[0] = *reinterpret_cast<const ink_v4u*>(p);
  c.u[1] = *reinterpret_cast<const ink_v4u*>(p + 8);
  return c.v;
}
"""
_ATTENTION_HIP = """
// A workgroup takes BQ queries of the G query heads that share a KV head, a
// wave per 16 queries of one head, so each K/V tile is read once for all of
// them. Q.K runs in bf16 as stored; the weights and V go to f16 for P.V. The
// next BK keys load into registers while this tile is used, then into the
// other LDS buffer, and the running softmax rescales once per tile.
constexpr int G = HQ / HKV, WQ = BQ / 16, NT = 32 * G * WQ;
constexpr int LDK = D + 8, LDV = BK + 8;
constexpr int PIECES = BK * D / 8, EACH = (PIECES + NT - 1) / NT;
constexpr float LOG2E = 1.4426950408889634f;
__shared__ __attribute__((aligned(16))) unsigned short ksb[2][BK * LDK];
__shared__ __attribute__((aligned(16))) _Float16 vtb[2][D * LDV];
const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
const int sub = lane & 15, half = lane >> 4;
const int hk = blockIdx.y, h = hk * G + wave / WQ, wq = wave % WQ;
const int L = dims[0], S = dims[1];
const int q0 = blockIdx.x * BQ;
const int qrow = q0 + wq * 16 + sub;
const bool qlive = qrow < L;
const int a = S - L + qrow;
const float qscale = scale[0] * LOG2E;
const unsigned short* qb = reinterpret_cast<const unsigned short*>(q);
const unsigned short* rb = reinterpret_cast<const unsigned short*>(rel);
const unsigned short* kbase = reinterpret_cast<const unsigned short*>(k) + (size_t)hk * S * D;
const unsigned short* vbase = reinterpret_cast<const unsigned short*>(v) + (size_t)hk * S * D;

// This lane's query, 16 dims a step.
ink_v16s qf[D / 16];
const unsigned short* qp = qb + ((size_t)(qlive ? qrow : 0) * HQ + h) * D;
#pragma unroll
for (int s16 = 0; s16 < D / 16; ++s16) qf[s16] = ink_row16(qp + s16 * 16);
ink_v8f o[D / 16];
#pragma unroll
for (int t = 0; t < D / 16; ++t) o[t] = (ink_v8f){};
float m = qlive ? sink[(size_t)qrow * HQ + h] * LOG2E : 0.f;
float l = 1.f;
const unsigned short* relq = rb + ((size_t)(qlive ? qrow : 0) * HQ + h) * R;

const int a_first = S - L + q0;
const int a_last = S - L + min(q0 + BQ, L) - 1;
const int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
const int k_begin = (lo / 16) * 16;
const int w_first = S - L + q0 + wq * 16;
const int w_last = min(w_first + 15, S - 1);
const bool wave_on = q0 + wq * 16 < L;

ink_v4u kreg[EACH], vreg[EACH];
auto fetch = [&](int k0) {
#pragma unroll
  for (int j = 0; j < EACH; ++j) {
    const int piece = tid + j * NT;
    const int row = piece / (D / 8), col = (piece % (D / 8)) * 8;
    const int key = k0 + row;
    if (piece < PIECES && key < S) {
      kreg[j] = *reinterpret_cast<const ink_v4u*>(kbase + (size_t)key * D + col);
      vreg[j] = *reinterpret_cast<const ink_v4u*>(vbase + (size_t)key * D + col);
    } else {
      kreg[j] = (ink_v4u){0u, 0u, 0u, 0u};
      vreg[j] = (ink_v4u){0u, 0u, 0u, 0u};
    }
  }
};
auto put = [&](int buf) {
  unsigned short* ks = ksb[buf];
  _Float16* vt = vtb[buf];
#pragma unroll
  for (int j = 0; j < EACH; ++j) {
    const int piece = tid + j * NT;
    if (piece < PIECES) {
      const int row = piece / (D / 8), col = (piece % (D / 8)) * 8;
      *reinterpret_cast<ink_v4u*>(&ks[row * LDK + col]) = kreg[j];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const unsigned word = vreg[j][w];
        vt[(col + 2 * w) * LDV + row] = (_Float16)__builtin_bit_cast(float, word << 16);
        vt[(col + 2 * w + 1) * LDV + row] = (_Float16)__builtin_bit_cast(float, word & 0xffff0000u);
      }
    }
  }
};

fetch(k_begin);
put(0);
__syncthreads();
for (int k0 = k_begin, it = 0; k0 <= a_last; k0 += BK, ++it) {
  const bool more = k0 + BK <= a_last;
  if (more) fetch(k0 + BK);
  const unsigned short* ks = ksb[it & 1];
  const _Float16* vt = vtb[it & 1];
  if (wave_on) {
    float x[BK / 16][8];
    bool on[BK / 16];
    float mx = -INFINITY;
#pragma unroll
    for (int g = 0; g < BK / 16; ++g) {
      const int g0 = k0 + g * 16;
      on[g] = g0 <= w_last && (WINDOW == 0 || g0 + 15 >= w_first - WINDOW + 1);
      if (on[g]) {
        ink_v8f s = (ink_v8f){};
        const unsigned short* kr = ks + (g * 16 + sub) * LDK;
#pragma unroll
        for (int s16 = 0; s16 < D / 16; ++s16) s = ink_wmma_bf(ink_row16(kr + s16 * 16), qf[s16], s);
        // Element i: key g0 + 2 i + half of this lane's query. Keys every
        // query of the wave sees, past the relative band, take no mask.
        const bool plain = g0 + 15 <= w_first - R && (WINDOW == 0 || g0 >= w_last - WINDOW + 1);
        if (plain) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            x[g][i] = s[i] * qscale;
            mx = fmaxf(mx, x[g][i]);
          }
        } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int key = g0 + 2 * i + half;
          const int d = a - key;
          const bool vis = qlive && key < S && d >= 0 && (WINDOW == 0 || d < WINDOW);
          float xi = s[i] * qscale;
          if (vis) {
            if (d < R) xi += ink_bf(relq[d]) * LOG2E;
          } else {
            xi = -INFINITY;
          }
          x[g][i] = xi;
          mx = fmaxf(mx, xi);
        }
        }
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) x[g][i] = -INFINITY;
      }
    }
    mx = fmaxf(mx, ink_other_half(mx));
    // The running max moves only when the tile's exceeds it by 2^8: the
    // weights stay below 256 (fine in fp16) and most tiles skip the rescale.
    const float n = mx > m + 8.f ? mx : m;
    const float f = exp2f(m - n);
    float sum = 0.f;
#pragma unroll
    for (int g = 0; g < BK / 16; ++g) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        x[g][i] = exp2f(x[g][i] - n);
        sum += x[g][i];
      }
    }
    sum += ink_other_half(sum);
    l = l * f + sum;
    m = n;
    if (__builtin_amdgcn_ballot_w32(f != 1.f)) {
#pragma unroll
      for (int t = 0; t < D / 16; ++t) o[t] = o[t] * f;
    }
#pragma unroll
    for (int g = 0; g < BK / 16; ++g) {
      if (!on[g]) continue;
      // The query's 16 weights in key order: 2 i + half here, the rest 16 lanes away.
      ink_v16h pf;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float other = ink_other_half(x[g][i]);
        pf[2 * i + half] = (_Float16)x[g][i];
        pf[2 * i + 1 - half] = (_Float16)other;
      }
#pragma unroll
      for (int t = 0; t < D / 16; ++t) {
        union { ink_v16h v; ink_v4u u[2]; } vf;
        const _Float16* vr = vt + (t * 16 + sub) * LDV + g * 16;
        vf.u[0] = *reinterpret_cast<const ink_v4u*>(vr);
        vf.u[1] = *reinterpret_cast<const ink_v4u*>(vr + 8);
        o[t] = ink_wmma(vf.v, pf, o[t]);
      }
    }
  }
  // The other buffer was last read before the previous barrier.
  if (more) {
    put((it & 1) ^ 1);
    __syncthreads();
  }
}
if (qlive) {
  const float inv = 1.f / l;
  unsigned short* ob = reinterpret_cast<unsigned short*>(out) + ((size_t)qrow * HQ + h) * D;
#pragma unroll
  for (int t = 0; t < D / 16; ++t) {
    float vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) vals[i] = o[t][i];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const unsigned u = __builtin_bit_cast(unsigned, vals[i] * inv);
      ob[t * 16 + 2 * i + half] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
  }
}
"""


# NVFP4 experts on RDNA 3 WMMA, for ROCm: rows sorted by expert, one wave a
# (16-row tile, four 16-column tiles). A group of 16 inputs is one WMMA step
# of the rows and the E2M1 codes (exact in bf16), times the column's E4M3
# scale, added in float32; the sum rounds to bf16 once, as gather_qmm's does.
# The codes are held [E, N / 16, K / 16, 16 columns, 2 words] and the scales
# [E, N / 16, K / 16, 16 columns], as a wave reads them.
_NVFP4_HEADER = """
typedef short ink_v16s __attribute__((ext_vector_type(16)));
typedef float ink_v8f __attribute__((ext_vector_type(8)));
typedef unsigned int ink_v4u __attribute__((ext_vector_type(4)));
__device__ __forceinline__ ink_v8f ink_wmma_bf16(ink_v16s a, ink_v16s b, ink_v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
// The bf16 bytes of code * scale for the eight E2M1 magnitudes, as byte
// tables: [low bytes 0-3, low 4-7, high 0-3, high 4-7]. The products have at
// most five significant bits, so the truncation to bf16 is exact.
__device__ __forceinline__ float ink_e4m3(unsigned b);
__device__ __forceinline__ void ink_fp4_table(unsigned b, unsigned* t) {
  const float scale = ink_e4m3(b);
  const unsigned f0 = 0u, f1 = __builtin_bit_cast(unsigned, 0.5f * scale),
                 f2 = __builtin_bit_cast(unsigned, scale),
                 f3 = __builtin_bit_cast(unsigned, 1.5f * scale),
                 f4 = __builtin_bit_cast(unsigned, 2.f * scale),
                 f5 = __builtin_bit_cast(unsigned, 3.f * scale),
                 f6 = __builtin_bit_cast(unsigned, 4.f * scale),
                 f7 = __builtin_bit_cast(unsigned, 6.f * scale);
  // [low of i, low of i + 1, high of i, high of i + 1]
  const unsigned a01 = __builtin_amdgcn_perm(f1, f0, 0x07030602u);
  const unsigned a23 = __builtin_amdgcn_perm(f3, f2, 0x07030602u);
  const unsigned a45 = __builtin_amdgcn_perm(f5, f4, 0x07030602u);
  const unsigned a67 = __builtin_amdgcn_perm(f7, f6, 0x07030602u);
  t[0] = __builtin_amdgcn_perm(a23, a01, 0x05040100u);
  t[1] = __builtin_amdgcn_perm(a67, a45, 0x05040100u);
  t[2] = __builtin_amdgcn_perm(a23, a01, 0x07060302u);
  t[3] = __builtin_amdgcn_perm(a67, a45, 0x07060302u);
}
// Eight E2M1 codes of a word as bf16 code * scale, in element order: a table
// lookup for the low and high byte of each, the sign, then interleaving.
__device__ __forceinline__ void ink_fp4_word(unsigned x, const unsigned* t, unsigned* out) {
  const unsigned m_even = x & 0x07070707u, m_odd = (x >> 4) & 0x07070707u;
  const unsigned l_even = __builtin_amdgcn_perm(t[1], t[0], m_even);
  const unsigned h_even = __builtin_amdgcn_perm(t[3], t[2], m_even) ^ ((x << 4) & 0x80808080u);
  const unsigned l_odd = __builtin_amdgcn_perm(t[1], t[0], m_odd);
  const unsigned h_odd = __builtin_amdgcn_perm(t[3], t[2], m_odd) ^ (x & 0x80808080u);
  // [low, high] of the even codes 0, 2 and 4, 6; then of the odd codes.
  const unsigned e01 = __builtin_amdgcn_perm(h_even, l_even, 0x05010400u);
  const unsigned e23 = __builtin_amdgcn_perm(h_even, l_even, 0x07030602u);
  const unsigned o01 = __builtin_amdgcn_perm(h_odd, l_odd, 0x05010400u);
  const unsigned o23 = __builtin_amdgcn_perm(h_odd, l_odd, 0x07030602u);
  out[0] = __builtin_amdgcn_perm(o01, e01, 0x05040100u);
  out[1] = __builtin_amdgcn_perm(o01, e01, 0x07060302u);
  out[2] = __builtin_amdgcn_perm(o23, e23, 0x05040100u);
  out[3] = __builtin_amdgcn_perm(o23, e23, 0x07060302u);
}
__device__ __forceinline__ ink_v16s ink_fp4(const unsigned* p, const unsigned* t) {
  union { ink_v16s f; unsigned u[8]; } c;
  ink_fp4_word(p[0], t, c.u);
  ink_fp4_word(p[1], t, c.u + 4);
  return c.f;
}
__device__ __forceinline__ float ink_e4m3(unsigned b) {
  const unsigned e = (b >> 3) & 15u, m = b & 7u;
  const float mag = e ? __builtin_bit_cast(float, ((e + 120u) << 23) | (m << 20))
                      : float(m) * 0.001953125f;
  return (b & 0x80u) ? -mag : mag;
}
__device__ __forceinline__ ink_v16s ink_load16(const unsigned short* p) {
  union { ink_v16s f; ink_v4u u[2]; } c;
  c.u[0] = *reinterpret_cast<const ink_v4u*>(p);
  c.u[1] = *reinterpret_cast<const ink_v4u*>(p + 8);
  return c.f;
}
"""
_NVFP4_EXPERTS = """
// A workgroup of WAVES waves takes RT 16-row tiles of one expert and WAVES
// adjacent groups of NG 16-column tiles. The rows go through LDS a KC-wide
// slice at a time, loaded once for all the waves, the next slice in flight
// while this one is used.
constexpr int KS = K / 16, KC = 64, NC = K / KC, ROWS = 16 * RT;
constexpr int PITCH = KC + 8;  // halves; staggers the rows across LDS banks
constexpr int THREADS = 32 * WAVES, PIECES = ROWS * KC / 8;
constexpr int EACH = (PIECES + THREADS - 1) / THREADS;
__shared__ __attribute__((aligned(16))) unsigned short lds[2][ROWS * PITCH];
const int tid = threadIdx.x, lane = tid & 31, sub = lane & 15, half = lane >> 4;
const int og0 = (blockIdx.y * WAVES + (tid >> 5)) * NG;
const bool cols = og0 < N / 16;
// Workgroup x takes row tile x of the experts' tiles laid end to end:
// first[e] is expert e's first tile, the extra x past the last tile none.
const int t = blockIdx.x;
if (t >= first[E]) return;
int lo = 0, hi = E;
while (hi - lo > 1) {
  const int mid = (lo + hi) >> 1;
  if (first[mid] <= t) lo = mid; else hi = mid;
}
const int e = lo;
const int end = offsets[e + 1];
const int r0 = offsets[e] + (t - first[e]) * ROWS;
const int tiles = min(RT, (end - r0 + 15) / 16);
const unsigned short* xs = reinterpret_cast<const unsigned short*>(x);
ink_v4u stage[EACH];
// The x row of each piece this thread loads: with GATHER, sorted row r
// reads token row src[r], so the rows need no gathered copy.
size_t xrow[EACH];
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int row = min(r0 + (tid + j * THREADS) / (KC / 8), end - 1);
  xrow[j] = GATHER ? (size_t)src[row] : (size_t)row;
}
auto fetch = [&](int c) {
#pragma unroll
  for (int j = 0; j < EACH; ++j) {
    const int piece = tid + j * THREADS;
    if (piece < PIECES) {
      const int col = (piece % (KC / 8)) * 8;
      stage[j] = *reinterpret_cast<const ink_v4u*>(xs + xrow[j] * K + c * KC + col);
    }
  }
};
auto put = [&](int buf) {
#pragma unroll
  for (int j = 0; j < EACH; ++j) {
    const int piece = tid + j * THREADS;
    if (piece < PIECES) {
      const int r = piece / (KC / 8), col = (piece % (KC / 8)) * 8;
      *reinterpret_cast<ink_v4u*>(&lds[buf][r * PITCH + col]) = stage[j];
    }
  }
};
ink_v8f acc[RT][NG];
const unsigned* wc[NG];
const unsigned char* sc[NG];
#pragma unroll
for (int ogi = 0; ogi < NG; ++ogi) {
#pragma unroll
  for (int rt = 0; rt < RT; ++rt) acc[rt][ogi] = (ink_v8f){};
  const size_t tile = size_t(e) * (N / 16) + (cols ? og0 + ogi : 0);
  wc[ogi] = w + tile * NC * 128 + sub * 8;
  sc[ogi] = s + tile * NC * 64 + sub * 4;
}
fetch(0);
put(0);
__syncthreads();
for (int c = 0; c < NC; ++c) {
  const int buf = c & 1;
  if (c + 1 < NC) fetch(c + 1);
  if (cols) {
    // A lane's codes and scales of a tile for the whole slice: eight words
    // (two per 16 inputs) and four scales.
    ink_v4u code[NG][2];
    unsigned scale[NG];
#pragma unroll
    for (int ogi = 0; ogi < NG; ++ogi) {
      code[ogi][0] = *reinterpret_cast<const ink_v4u*>(wc[ogi] + c * 128);
      code[ogi][1] = *reinterpret_cast<const ink_v4u*>(wc[ogi] + c * 128 + 4);
      scale[ogi] = *reinterpret_cast<const unsigned*>(sc[ogi] + c * 64);
    }
#pragma unroll
    for (int kk = 0; kk < KC / 16; ++kk) {
      // A lane's column of a tile shares one scale: the codes decode straight
      // to code * scale, exact in bf16. Each decoded tile serves every row tile.
      ink_v16s b[NG];
#pragma unroll
      for (int ogi = 0; ogi < NG; ++ogi) {
        unsigned table[4], words[2];
        ink_fp4_table((scale[ogi] >> (8 * kk)) & 0xffu, table);
        words[0] = code[ogi][kk >> 1][(kk & 1) * 2];
        words[1] = code[ogi][kk >> 1][(kk & 1) * 2 + 1];
        b[ogi] = ink_fp4(words, table);
      }
#pragma unroll
      for (int rt = 0; rt < RT; ++rt) {
        if (rt < tiles) {
          const ink_v16s a = ink_load16(&lds[buf][(rt * 16 + sub) * PITCH + kk * 16]);
#pragma unroll
          for (int ogi = 0; ogi < NG; ++ogi) acc[rt][ogi] = ink_wmma_bf16(a, b[ogi], acc[rt][ogi]);
        }
      }
    }
  }
  if (c + 1 < NC) {
    put(buf ^ 1);
    __syncthreads();
  }
}
if (!cols) return;
unsigned short* yb = reinterpret_cast<unsigned short*>(y);
#pragma unroll
for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
  for (int ogi = 0; ogi < NG; ++ogi) {
    // Element i of the accumulator is row 2 i + half; copied out first, as an
    // indexed read of the vector inside the guarded loop read element 0 only.
    float vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) vals[i] = acc[rt][ogi][i];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int r = r0 + rt * 16 + 2 * i + half;
      if (r < end) {
        const unsigned u = __builtin_bit_cast(unsigned, vals[i]);
        yb[(size_t)r * N + (og0 + ogi) * 16 + sub] =
            (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
      }
    }
  }
}
"""


# A workgroup's 16-row tiles, each wave's 16-column tiles, and its waves. A
# decoded weight tile serves every row tile, so more row tiles spend less of
# the VALU (which also issues the WMMA) on decoding; on gfx1151 six row tiles
# of two column tiles, eight waves to a workgroup, beat 2 x 4 by a fifth.
_NVFP4_ROW_TILES = 6
_NVFP4_COL_TILES = 2
_NVFP4_WAVES = 8


@functools.cache
def _nvfp4_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_nvfp4_experts",
        input_names=["x", "w", "s", "offsets", "first", "src"],
        output_names=["y"],
        source=_NVFP4_EXPERTS,
        header=_NVFP4_HEADER,
    )


def nvfp4_fits(bank) -> bool:
    """Whether the ROCm NVFP4 expert kernel takes this bank."""
    weight = bank.weight
    if bank.quant.get("mode") != "nvfp4" or weight.ndim != 3:
        return False
    inputs = weight.shape[2] * 8
    return weight.shape[1] % (16 * _NVFP4_COL_TILES) == 0 and inputs % 64 == 0


def nvfp4_pack(bank) -> tuple[mx.array, mx.array]:
    """(codes, scales) of an NVFP4 bank in the order the kernel reads them:
    per 16-column tile and 64 inputs, each column's eight code words and
    four scales together."""
    count, width, words = bank.weight.shape
    codes = bank.weight.reshape(count, width // 16, 16, words // 8, 8)
    codes = mx.contiguous(codes.transpose(0, 1, 3, 2, 4))
    groups = bank.scales.shape[-1]
    scales = bank.scales.reshape(count, width // 16, 16, groups // 4, 4)
    return codes, mx.contiguous(scales.transpose(0, 1, 3, 2, 4))


def nvfp4_tile_rows() -> int:
    """Rows of the expert kernel's row tile: ``dispatch`` lays tiles out."""
    return 16 * _NVFP4_ROW_TILES


def nvfp4_gemm(
    x: mx.array, packed, experts: mx.array, route=None, rows=None, gather: bool = True
) -> mx.array:
    """``x`` [R, K] bf16 rows sorted by expert, each times its expert of a
    packed NVFP4 bank: [R, N] bf16, before the bank's tensor scale.

    ``route``: ``dispatch``'s (offsets, first, source) for ``rows`` sorted
    rows; with ``gather`` ``x`` is the token rows and sorted row r reads row
    ``source[r]``, else ``x`` is already in expert order.
    """
    codes, scales = packed
    count, tiles = codes.shape[0], codes.shape[1]
    inputs = x.shape[1]
    waves = _NVFP4_WAVES
    tile_rows = nvfp4_tile_rows()
    if route is None:
        rows = x.shape[0]
        zero = mx.zeros((1,), dtype=mx.int32)
        counts = mx.zeros((count,), dtype=mx.int32).at[experts.astype(mx.int32)].add(1)
        offsets = mx.concatenate([zero, mx.cumsum(counts)])
        # Each expert's rows in whole tiles; the grid covers the most there
        # can be without reading the counts back.
        first = mx.concatenate([zero, mx.cumsum((counts + tile_rows - 1) // tile_rows)])
        src = zero
    else:
        offsets, first, src = route
    blocks = -(-rows // tile_rows) + count
    width = tiles * 16
    return _nvfp4_kernel()(
        inputs=[mx.contiguous(x), codes, scales, offsets, first, src],
        template=[
            ("K", inputs),
            ("N", width),
            ("E", count),
            ("RT", _NVFP4_ROW_TILES),
            ("NG", _NVFP4_COL_TILES),
            ("WAVES", waves),
            ("GATHER", int(route is not None and gather)),
        ],
        grid=(blocks * 32 * waves, -(-width // (16 * _NVFP4_COL_TILES * waves)), 1),
        threadgroup=(32 * waves, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[mx.bfloat16],
    )[0]


# The routed rows of a chunk laid out by expert, in one workgroup: counts in
# LDS, the expert offsets and row-tile starts, then each (token, slot) entry
# takes the next row of its expert. Rows of an expert come in any order;
# every row of the expert GEMMs depends on its own input row only. The
# weights take the down bank's NVFP4 tensor scale (POST).
_DISPATCH_HIP = """
__shared__ int count[E], cursor[E], base[E + 1], tiles[E + 1];
const int tid = threadIdx.x, n = dims[0];
for (int e = tid; e < E; e += blockDim.x) {
  count[e] = 0;
  cursor[e] = 0;
}
__syncthreads();
for (int i = tid; i < n; i += blockDim.x) atomicAdd(&count[inds[i]], 1);
__syncthreads();
if (tid == 0) {
  int o = 0, f = 0;
  for (int e = 0; e < E; ++e) {
    base[e] = o;
    tiles[e] = f;
    o += count[e];
    f += (count[e] + TR - 1) / TR;
  }
  base[E] = o;
  tiles[E] = f;
}
__syncthreads();
for (int e = tid; e <= E; e += blockDim.x) {
  offsets[e] = base[e];
  first[e] = tiles[e];
}
for (int i = tid; i < n; i += blockDim.x) {
  const int e = inds[i];
  const int pos = base[e] + atomicAdd(&cursor[e], 1);
  source[pos] = i / TOPK;
  sorted[pos] = e;
  inverse[i] = pos;
  scaled[i] = POST ? weights[i] * post[e] : weights[i];
}
"""


@functools.cache
def _dispatch_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_dispatch_hip",
        input_names=["inds", "weights", "post", "dims"],
        output_names=["offsets", "first", "source", "sorted", "inverse", "scaled"],
        source=_DISPATCH_HIP,
    )


def dispatch(inds: mx.array, weights: mx.array, experts: int, post=None):
    """Routed entries by expert, for the expert kernels: (offsets [E+1],
    tile starts [E+1], token of each sorted row, expert of each sorted row,
    sorted row of each entry, weights times ``post`` of their expert)."""
    tokens, each = inds.shape
    n = tokens * each
    return _dispatch_kernel()(
        inputs=[
            inds.reshape(-1).astype(mx.int32),
            weights.reshape(-1).astype(mx.float32),
            mx.ones((1,), dtype=mx.float32) if post is None else post,
            mx.array([n], dtype=mx.int32),
        ],
        template=[("E", experts), ("TOPK", each), ("TR", nvfp4_tile_rows()), ("POST", int(post is not None))],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(experts + 1,), (experts + 1,), (n,), (n,), (n,), (tokens, each)],
        output_dtypes=[mx.int32, mx.int32, mx.int32, mx.uint32, mx.uint32, mx.float32],
    )


# Keys of a ROCm attention tile, and queries of a workgroup per query head:
# a full layer reads its long past once for more queries, a sliding one
# keeps more workgroups in flight. Larger key tiles hold fewer workgroups.
_ATTENTION_KEYS = 16
_ATTENTION_QUERIES = {True: 128, False: 64}


@functools.cache
def _attention_hip_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_attention_hip",
        input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
        output_names=["out"],
        source=_ATTENTION_HIP,
        header=_ATTENTION_HIP_HEADER,
    )


@functools.cache
def _attention_kernel():
    return _cuda_kernel(
        name="omlx_inkling2_attention",
        input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
        output_names=["out"],
        source=_ATTENTION,
        header=_ATTENTION_HEADER,
    )


def covers_attention(q: mx.array, extent: int) -> bool:
    """Whether the attention kernel takes these queries [1, L, H, 128]."""
    return q.dtype == mx.bfloat16 and q.shape[0] == 1 and q.shape[-1] == 128


def attention(q, keys, values, rel, sink, scale: float, window: int) -> mx.array:
    """Banded attention with per-query sinks: q [1, L, Hq, D], keys/values
    [1, Hkv, S, D] (the queries are the last L keys), rel [1, L, Hq, R],
    sink [1, L, Hq]. Returns [1, L, Hq, D]."""
    _, length, heads, dim = q.shape
    kv_heads, total = keys.shape[1], keys.shape[2]
    # A view of the leading axis: x[0] runs a gather on ROCm.
    lead = functools.partial(mx.squeeze, axis=0)
    hip = _rocm()
    kernel = _attention_hip_kernel() if hip else _attention_kernel()
    group = heads // kv_heads
    block = _ATTENTION_QUERIES[window == 0]
    out = kernel(
        inputs=[
            mx.contiguous(lead(q)),
            mx.contiguous(lead(keys)),
            mx.contiguous(lead(values)),
            mx.contiguous(lead(rel)),
            mx.contiguous(lead(sink).astype(mx.float32)),
            mx.array([length, total], dtype=mx.int32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("D", dim),
            ("HQ", heads),
            ("HKV", kv_heads),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
            *((("BQ", block), ("BK", _ATTENTION_KEYS)) if hip else ()),
        ],
        grid=(
            (-(-length // block) * 32 * group * (block // 16), kv_heads, 1)
            if hip
            else (-(-length // 64) * 128, heads, 1)
        ),
        threadgroup=(32 * group * (block // 16), 1, 1) if hip else (128, 1, 1),
        output_shapes=[(length, heads, dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out[None]


# One 256-thread block per token row, as the host's Metal kernel: dot products
# of every source with every query, softmax over the sources, weighted sum.
_MIX = """
constexpr int TG = 256;
const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
const size_t base = (size_t)row * H;
float acc[NS * Q];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) acc[i] = 0.f;
for (int c = tid; c < H; c += TG) {
  float s[NS];
  LOADS
  #pragma unroll
  for (int qi = 0; qi < Q; ++qi) {
    const float qv = query[qi * H + c];
    #pragma unroll
    for (int n = 0; n < NS; ++n) acc[n * Q + qi] = fmaf(s[n], qv, acc[n * Q + qi]);
  }
}
__shared__ float red[NS * Q][TG / 32];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) {
  float v = acc[i];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
  if (lane == 0) red[i][warp] = v;
}
__syncthreads();
float w[NS * Q];
#pragma unroll
for (int i = 0; i < NS * Q; ++i) {
  float t = 0.f;
  #pragma unroll
  for (int j = 0; j < TG / 32; ++j) t += red[i][j];
  w[i] = t * scale[0];
}
#pragma unroll
for (int qi = 0; qi < Q; ++qi) {
  float m = -INFINITY;
  #pragma unroll
  for (int n = 0; n < NS; ++n) m = fmaxf(m, w[n * Q + qi]);
  float z = 0.f;
  #pragma unroll
  for (int n = 0; n < NS; ++n) {
    w[n * Q + qi] = expf(w[n * Q + qi] - m);
    z += w[n * Q + qi];
  }
  #pragma unroll
  for (int n = 0; n < NS; ++n) w[n * Q + qi] /= z;
}
for (int c = tid; c < H; c += TG) {
  float s[NS];
  LOADS
  #pragma unroll
  for (int qi = 0; qi < Q; ++qi) {
    float y = w[qi] * s[0];
    #pragma unroll
    for (int n = 1; n < NS; ++n) y = fmaf(w[n * Q + qi], s[n], y);
    out[((size_t)row * Q + qi) * H + c] = y;
  }
}
"""


@functools.cache
def _mix_kernel(sources: int, queries: int):
    names = [f"b{i}" for i in range(sources - 1)] + ["p"]
    loads = " ".join(f"s[{i}] = {n}[base + c];" for i, n in enumerate(names))
    return _cuda_kernel(
        name=f"omlx_inkling2_mix_{sources}_{queries}",
        input_names=[*names, "query", "scale"],
        output_names=["out"],
        source=_MIX.replace("LOADS", loads),
    )


def mix(blocks: list, partial: mx.array, query: mx.array, scale: float) -> mx.array:
    """Depth mixture of unit-RMS float32 sources [1, L, H] with ``query``
    [Q, H]: [1, L, Q, H] float32."""
    _, length, hidden = partial.shape
    queries = query.shape[0]
    sources = len(blocks) + 1
    return _mix_kernel(sources, queries)(
        inputs=[*blocks, partial, query, mx.array([scale], dtype=mx.float32)],
        template=[("NS", sources), ("Q", queries), ("H", hidden)],
        grid=(length * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, queries, hidden)],
        output_dtypes=[mx.float32],
    )[0]


# One warp per token row: sigmoid scores, top-k by score + bias (the lowest
# expert wins a tie), weights renormalized from the unbiased scores.
_ROUTE = """
const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
const int lane = threadIdx.x & 31;
if (row >= rows[0]) return;
constexpr int PER = E / 32;
float sc[PER], key[PER];
#pragma unroll
for (int j = 0; j < PER; ++j) {
  const int e = lane + 32 * j;
  const float s = 1.f / (1.f + expf(-logits[(size_t)row * E + e]));
  sc[j] = s;
  key[j] = s + bias[e];
}
float total = 0.f;
for (int t = 0; t < K; ++t) {
  float best = -INFINITY;
  int bj = 0;
  #pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (key[j] > best) {
      best = key[j];
      bj = j;
    }
  }
  float top = best;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, off));
  uint32_t mine = best == top ? (uint32_t)(lane + 32 * bj) : 0xFFFFFFFFu;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) mine = min(mine, __shfl_xor_sync(0xffffffff, mine, off));
  float s = 0.f;
  #pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (lane == (int)(mine % 32) && j == (int)(mine / 32)) {
      s = sc[j];
      key[j] = -INFINITY;
    }
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
  total += s;
  if (lane == 0) {
    inds[(size_t)row * K + t] = mine;
    weights[(size_t)row * K + t] = s;
  }
}
if (lane == 0) {
  const float norm = route_scale[0] / (total + 1e-20f);
  for (int t = 0; t < K; ++t) weights[(size_t)row * K + t] *= norm;
}
"""


@functools.cache
def _route_kernel():
    return _cuda_kernel(
        name="omlx_inkling2_route",
        input_names=["logits", "bias", "route_scale", "rows"],
        output_names=["inds", "weights"],
        source=_ROUTE,
    )


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """([N, K] uint32 experts, [N, K] float32 weights) of float32 logits."""
    rows, experts = logits.shape
    warps = 8
    return _route_kernel()(
        inputs=[
            logits,
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
            mx.array([rows], dtype=mx.int32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(-(-rows // warps) * warps * 32, 1, 1),
        threadgroup=(warps * 32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )


# The MiniMax router kernel's sums (each logit adds its K products in float32
# FMAs, in ``runs`` parts summed in order) for ROWS token rows per block, so
# every router column loads once for all of them.
_ROUTER = """
extern __shared__ __align__(16) unsigned char smem[];
__nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem);
const int K = x_shape[1], outputs = w_shape[1], rows = x_shape[0];
const int row0 = blockIdx.x * ROWS, e = threadIdx.x;
for (int i = threadIdx.x; i < ROWS * K; i += blockDim.x) {
  const int r = row0 + i / K;
  xs[i] = r < rows ? x[(size_t)r * K + i % K] : __float2bfloat16(0.f);
}
__syncthreads();
if (e >= outputs) return;
const int parts = runs[0], span = (K / 16 / parts) * 16;
float total[ROWS];
for (int p = 0; p < parts; ++p) {
  const int end = p + 1 == parts ? K : (p + 1) * span;
  float acc[ROWS];
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
  for (int k = p * span; k < end; ++k) {
    const float wk = w[(size_t)k * outputs + e];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = fmaf(__bfloat162float(xs[r * K + k]), wk, acc[r]);
  }
  #pragma unroll
  for (int r = 0; r < ROWS; ++r) total[r] = p == 0 ? acc[r] : total[r] + acc[r];
}
#pragma unroll
for (int r = 0; r < ROWS; ++r)
  if (row0 + r < rows) out[(size_t)(row0 + r) * outputs + e] = total[r];
"""
_ROUTER_ROWS = 4


@functools.cache
def _router_kernel(shared: int):
    return _cuda_kernel(
        name="omlx_inkling2_router",
        input_names=["x", "w", "runs"],
        output_names=["out"],
        source=_ROUTER,
        shared_memory=shared,
    )


def router_logits(x: mx.array, columns: mx.array) -> mx.array:
    """Float32 router logits [rows, experts] of BF16 ``x`` [rows, K] and the
    float32 router weight as ``columns`` [K, experts], in the host's order."""
    rows, inputs = x.shape
    outputs = columns.shape[1]
    threads = -(-outputs // 32) * 32
    runs = mx.array([minimax.host_runs(rows, outputs, inputs)], dtype=mx.int32)
    return _router_kernel(_ROUTER_ROWS * inputs * 2)(
        inputs=[x, columns, runs],
        template=[("ROWS", _ROUTER_ROWS)],
        grid=(threads * -(-rows // _ROUTER_ROWS), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, outputs)],
        output_dtypes=[mx.float32],
    )[0]


# MLX's own sigmoid and tanh, so the fused ops round as the plain ones do.
_MLX_OPS = """
#include "mlx/backend/cuda/device/unary_ops.cuh"
#include <cuda_bf16.h>
"""
# The NVFP4 post-scale (the product rounds to BF16) and the softcapped SwiGLU
# of the interleaved gate (even) and up (odd) columns, one output each.
_ACTIVATION = """
using namespace mlx::core::cu;
const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
const size_t inter = y_shape[1] / 2;
if (id >= (size_t)y_shape[0] * inter) return;
const size_t r = id / inter, i = id % inter;
__nv_bfloat16 g = y[r * 2 * inter + 2 * i], u = y[r * 2 * inter + 2 * i + 1];
if (POST) {
  const __nv_bfloat16 s = __float2bfloat16_rn(post[experts[r]]);
  g = __float2bfloat16_rn(__bfloat162float(g) * __bfloat162float(s));
  u = __float2bfloat16_rn(__bfloat162float(u) * __bfloat162float(s));
}
const float gf = __bfloat162float(g), uf = __bfloat162float(u);
const float gcap = caps[0], ucap = caps[1];
float out = Tanh{}(gf / gcap) * gcap;
out = out * Sigmoid{}(gf);
out = out * (Tanh{}(uf / ucap) * ucap);
act[id] = __float2bfloat16_rn(out);
"""


@functools.cache
def _activation_kernel():
    return _cuda_kernel(
        name="omlx_inkling2_activation",
        input_names=["y", "experts", "post", "caps"],
        output_names=["act"],
        source=_ACTIVATION,
        header=_MLX_OPS,
    )


def activation(y: mx.array, experts: mx.array, post, caps) -> mx.array:
    """``layers.Experts.activation`` of BF16 ``y`` [R, 2I] in one launch."""
    rows, width = y.shape
    count = rows * (width // 2)
    gcap, ucap = caps
    return _activation_kernel()(
        inputs=[
            y,
            experts.astype(mx.uint32),
            post if post is not None else mx.ones((1,), dtype=mx.float32),
            mx.concatenate([gcap.reshape(1), ucap.reshape(1)]),
        ],
        template=[("POST", int(post is not None))],
        grid=(-(-count // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, width // 2)],
        output_dtypes=[mx.bfloat16],
    )[0]


# ``layers.static_conv`` in one launch: the taps multiply and add separately
# in float32, round to the weight dtype, add the input and scale.
_CONV_HEADER = """
#include <cuda_bf16.h>
__device__ inline float ink_f(float x) { return x; }
__device__ inline float ink_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ inline void ink_put(float x, float* out) { *out = x; }
__device__ inline void ink_put(float x, __nv_bfloat16* out) { *out = __float2bfloat16_rn(x); }
__device__ inline float ink_round(float x, float) { return x; }
__device__ inline float ink_round(float x, __nv_bfloat16) {
  return __bfloat162float(__float2bfloat16_rn(x));
}
"""
_CONV = """
const int L = x_shape[1], C = x_shape[2];
const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
if (id >= (size_t)(L + TAPS - 1) * C) return;
const int t = id / C, c = id % C;
auto xp = [&](int j) -> float {
  return j < TAPS - 1 ? state[(size_t)j * C + c] : ink_f(x[(size_t)(j - TAPS + 1) * C + c]);
};
if (t >= L) {
  // The last TAPS - 1 rows of [state, x] are the next state.
  window[(size_t)(t - L) * C + c] = xp(t);
  return;
}
float acc = __fmul_rn(ink_f(w[(size_t)c * TAPS]), xp(t));
#pragma unroll
for (int j = 1; j < TAPS; ++j) acc = __fadd_rn(acc, __fmul_rn(ink_f(w[(size_t)c * TAPS + j]), xp(t + j)));
const float rounded = ink_round(acc, w[0]);
ink_put(__fmul_rn(__fadd_rn(rounded, xp(t + TAPS - 1)), scale[0]), y + id);
"""
_CONV_KERNELS: dict = {}


def static_conv(x, w, state, scale: float):
    """(``layers.static_conv`` output, the next state [1, K - 1, C])."""
    key = (x.dtype, w.dtype)
    kernel = _CONV_KERNELS.get(key)
    if kernel is None:
        # A kernel object compiles for the dtypes of its first call only.
        names = "_".join(str(d).rsplit(".", 1)[-1] for d in key)
        kernel = _CONV_KERNELS[key] = _cuda_kernel(
            name=f"omlx_inkling2_conv_{names}",
            input_names=["x", "w", "state", "scale"],
            output_names=["y", "window"],
            source=_CONV,
            header=_CONV_HEADER,
        )
    _, length, channels = x.shape
    taps = w.shape[-1]
    count = (length + taps - 1) * channels
    y, window = kernel(
        inputs=[x, w, state, mx.array([scale], dtype=mx.float32)],
        template=[("TAPS", taps)],
        grid=(-(-count // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, (1, taps - 1, channels)],
        output_dtypes=[x.dtype, mx.float32],
    )
    return y, window


def _format(bank) -> str | None:
    """The bank's carrier as the expert GEMMs name it, or None."""
    quant = bank.quant
    if quant["mode"] in _FORMATS:
        return quant["mode"] if bank.scales.dtype == mx.uint8 else None
    affine = (
        quant["mode"] == "affine"
        and quant["group_size"] == 64
        and quant["bits"] in _AFFINE_BITS
        and bank.biases is not None
        and bank.scales.dtype == mx.bfloat16
    )
    return f"affine{quant['bits']}" if affine else None


def covers_bank(bank, dtype: mx.Dtype) -> bool:
    """Whether an expert GEMM takes this bank for BF16 rows: a checked carrier
    in 128 tiles."""
    inputs = bank.weight.shape[2] * 32 // bank.quant["bits"]
    if dtype != mx.bfloat16 or inputs % 128 or bank.weight.shape[1] % 128:
        return False
    return f"experts:{_format(bank)}" in _on


def gather(x: mx.array, bank, experts: mx.array) -> mx.array:
    """``x`` [R, K] rows, sorted by expert, each through its expert of ``bank``."""
    if bank.quant["mode"] in _FORMATS:
        return minimax.gather(x, bank, experts)
    return deepseek.expert_down(x, experts, bank)


# The query, key and relative-state clips for HIP: one wave per (row, head)
# vector of D values at column OFF + head * D of rows W wide. In float32 the
# vector takes its per-head scale (SCALED), is clipped to an L2 norm of
# threshold and rounded to bf16, then (TAU) scaled by its row's tau and
# rounded again, as the plain ops round between the two.
_CLIP_HIP = """
const int v = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
const int lane = threadIdx.x & 31;
if (v >= dims[0] * HEADS) return;
const int row = v / HEADS, head = v % HEADS;
const unsigned short* xb = reinterpret_cast<const unsigned short*>(x) + (size_t)row * W + OFF + head * D;
constexpr int EACH = (D + 31) / 32;
float vals[EACH];
float ss = 0.f;
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int d = lane + 32 * j;
  float f = 0.f;
  if (d < D) {
    f = __builtin_bit_cast(float, unsigned(xb[d]) << 16);
    if (SCALED) f *= scale[head * D + d];
  }
  vals[j] = f;
  ss += f * f;
}
#pragma unroll
for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off);
const float c = fminf(threshold[0] / (sqrtf(ss) + 1e-6f), 1.f);
const float t = TAU ? tau[row] : 1.f;
unsigned short* ob = reinterpret_cast<unsigned short*>(out) + (size_t)v * D;
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int d = lane + 32 * j;
  if (d < D) {
    unsigned u = __builtin_bit_cast(unsigned, vals[j] * c);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    if (TAU) {
      u = __builtin_bit_cast(unsigned, __builtin_bit_cast(float, u) * t);
      u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    }
    ob[d] = (unsigned short)(u >> 16);
  }
}
"""

# The attention output gate for HIP: out * sigmoid(gate) in float32, rounded
# to bf16, the gate at column OFF of rows W wide; 8 values a thread.
_GATE_HIP = """
const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
const int n = dims[0];
if (i >= (size_t)n * N) return;
const size_t row = i / N, col = i % N;
const unsigned short* ob = reinterpret_cast<const unsigned short*>(attn) + i;
const unsigned short* gb = reinterpret_cast<const unsigned short*>(proj) + row * W + OFF + col;
unsigned short* yb = reinterpret_cast<unsigned short*>(y) + i;
#pragma unroll
for (int j = 0; j < 8; ++j) {
  const float a = __builtin_bit_cast(float, unsigned(ob[j]) << 16);
  const float g = __builtin_bit_cast(float, unsigned(gb[j]) << 16);
  const unsigned u = __builtin_bit_cast(unsigned, a * (1.f / (1.f + expf(-g))));
  yb[j] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
"""


@functools.cache
def _clip_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_clip_hip",
        input_names=["x", "scale", "threshold", "tau", "dims"],
        output_names=["out"],
        source=_CLIP_HIP,
    )


@functools.cache
def _gate_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_gate_hip",
        input_names=["attn", "proj", "dims"],
        output_names=["y"],
        source=_GATE_HIP,
    )


def l2_clip(base, offset, heads, dim, scale, threshold, tau=None) -> mx.array:
    """Heads ``[offset, offset + heads * dim)`` of ``base`` [1, L, W] bf16,
    clipped (and tau-scaled): [1, L, heads, dim] bf16."""
    length, width = base.shape[1], base.shape[2]
    one = mx.zeros((1,), dtype=mx.float32)
    out = _clip_kernel()(
        inputs=[
            base,
            one if scale is None else scale.astype(mx.float32).reshape(-1),
            mx.array([threshold], dtype=mx.float32),
            one if tau is None else tau.astype(mx.float32).reshape(-1),
            mx.array([length], dtype=mx.int32),
        ],
        template=[
            ("D", dim),
            ("HEADS", heads),
            ("W", width),
            ("OFF", offset),
            ("SCALED", int(scale is not None)),
            ("TAU", int(tau is not None)),
        ],
        grid=(-(-length * heads // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, heads, dim)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return out


def gate(attn: mx.array, base: mx.array, offset: int) -> mx.array:
    """``attn`` [1, L, N] bf16 times sigmoid of ``base``'s columns
    ``[offset, offset + N)``: [1, L, N] bf16."""
    length, width = attn.shape[1], attn.shape[2]
    count = length * width // 8
    return _gate_kernel()(
        inputs=[mx.contiguous(attn), base, mx.array([length], dtype=mx.int32)],
        template=[("N", width), ("W", base.shape[2]), ("OFF", offset)],
        grid=(-(-count // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, length, width)],
        output_dtypes=[mx.bfloat16],
    )[0]


# The loop controllers for HIP, around their two matmuls. Input i of n reads
# query (QX == 1 ? 0 : i) of ``mixed`` [L, QX, H] and message M0 + i of
# ``messages`` [L, 4, H], float32. ``inputs`` writes z [n, L, 2H] bf16: the
# unit-RMS input, then the unit-RMS message. ``outputs`` takes the gates
# [n, L, 2H] float32 (gain, then forget), forms (1 - tanh(forget)) * x +
# tanh(gain) * message and its RMSNorm with ``weight``, rounded to bf16.
# One 256-thread block per (input, row).
_MESSAGES_HEADER = """
__device__ __forceinline__ float ink_block_sum(float v, float* red) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) t += red[j];
  return t;
}
"""
_MESSAGES_IN = """
__shared__ float red[8];
const int row = blockIdx.x, i = blockIdx.y;
const float* xr = mixed + ((size_t)row * QX + (QX == 1 ? 0 : i)) * H;
const float* mr = messages + ((size_t)row * 4 + M0 + i) * H;
constexpr int EACH = (H + 255) / 256;
float xv[EACH], mv[EACH];
float sx = 0.f, sm = 0.f;
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int c = threadIdx.x + 256 * j;
  xv[j] = c < H ? xr[c] : 0.f;
  mv[j] = c < H ? mr[c] : 0.f;
  sx += xv[j] * xv[j];
  sm += mv[j] * mv[j];
}
sx = ink_block_sum(sx, red);
sm = ink_block_sum(sm, red);
const float rx = rsqrtf(sx / H + 1e-6f), rm = rsqrtf(sm / H + 1e-6f);
unsigned short* zr = reinterpret_cast<unsigned short*>(z) + ((size_t)i * L + row) * 2 * H;
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int c = threadIdx.x + 256 * j;
  if (c < H) {
    unsigned u = __builtin_bit_cast(unsigned, xv[j] * rx);
    zr[c] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    u = __builtin_bit_cast(unsigned, mv[j] * rm);
    zr[H + c] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
}
"""
_MESSAGES_OUT = """
__shared__ float red[8];
const int row = blockIdx.x, i = blockIdx.y;
const float* xr = mixed + ((size_t)row * QX + (QX == 1 ? 0 : i)) * H;
const float* mr = messages + ((size_t)row * 4 + M0 + i) * H;
const float* gr = gates + ((size_t)i * L + row) * 2 * H;
constexpr int EACH = (H + 255) / 256;
float yv[EACH];
float ss = 0.f;
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int c = threadIdx.x + 256 * j;
  float y = 0.f;
  if (c < H) {
    const float gain = tanhf(gr[c]), forget = tanhf(gr[H + c]);
    y = (1.f - forget) * xr[c] + gain * mr[c];
  }
  yv[j] = y;
  ss += y * y;
}
ss = ink_block_sum(ss, red);
const float r = rsqrtf(ss / H + EPS[0]);
unsigned short* out = reinterpret_cast<unsigned short*>(y) + ((size_t)i * L + row) * H;
const unsigned short* wb = reinterpret_cast<const unsigned short*>(weight);
#pragma unroll
for (int j = 0; j < EACH; ++j) {
  const int c = threadIdx.x + 256 * j;
  if (c < H) {
    const float wv = __builtin_bit_cast(float, unsigned(wb[c]) << 16);
    const unsigned u = __builtin_bit_cast(unsigned, yv[j] * r * wv);
    out[c] = (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
}
"""


@functools.cache
def _messages_kernels():
    return (
        mx.fast.cuda_kernel(
            name="omlx_inkling2_messages_in",
            input_names=["mixed", "messages"],
            output_names=["z"],
            source=_MESSAGES_IN,
            header=_MESSAGES_HEADER,
        ),
        mx.fast.cuda_kernel(
            name="omlx_inkling2_messages_out",
            input_names=["mixed", "messages", "gates", "weight", "EPS"],
            output_names=["y"],
            source=_MESSAGES_OUT,
            header=_MESSAGES_HEADER,
        ),
    )


def mix_messages_normed(mixed, messages, m0, n, down, gain_forget, weight, eps):
    """The loop controllers on ``n`` inputs and the RMSNorm after them:
    ``mixed`` [1, L, QX, H] float32 (QX 1: every input reads query 0),
    ``messages`` [1, L, 4, H]; ``down`` [n, 2H, r] bf16 (the x half, then the
    message half), ``gain_forget`` [n, r, 2H] float32. Returns [n, 1, L, H]
    bf16."""
    _, length, qx, hidden = mixed.shape
    first, second = _messages_kernels()
    template = [("H", hidden), ("L", length), ("QX", qx), ("M0", m0)]
    mixed = mx.contiguous(mixed)
    messages = mx.contiguous(messages)
    z = first(
        inputs=[mixed, messages],
        template=template,
        grid=(length * 256, n, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, length, 2 * hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]
    # One bf16 matmul over both halves accumulates in float32 and rounds
    # once, as the host's two float32 halves summed and rounded do.
    low = (z @ down).astype(mx.float32)
    gates = low @ gain_forget
    return second(
        inputs=[mixed, messages, gates, weight, mx.array([eps], dtype=mx.float32)],
        template=template,
        grid=(length * 256, n, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, 1, length, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


# The MiniMax combine (mlx-serve moe_prefill_sum) for HIP: each product
# rounds before its add, in expert order, then one rounding to bf16.
_WEIGHTED_SUM_HIP = """
const int t = blockIdx.x;
const int d = threadIdx.x * 8;
float acc[8];
#pragma unroll
for (int i = 0; i < 8; ++i) acc[i] = 0.f;
for (int j = 0; j < TOPK; ++j) {
  const unsigned r = inverse[t * TOPK + j];
  const float weight = weights[t * TOPK + j];
  const uint4 v = *reinterpret_cast<const uint4*>(
      reinterpret_cast<const unsigned short*>(down) + (size_t)r * H + d);
  const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float value = __builtin_bit_cast(
        float, i & 1 ? words[i / 2] & 0xffff0000u : words[i / 2] << 16);
    acc[i] = __fadd_rn(acc[i], __fmul_rn(value, weight));
  }
}
unsigned out[4];
#pragma unroll
for (int i = 0; i < 4; ++i) {
  unsigned lo = __builtin_bit_cast(unsigned, acc[2 * i]);
  unsigned hi = __builtin_bit_cast(unsigned, acc[2 * i + 1]);
  lo = (lo + 0x7fffu + ((lo >> 16) & 1u)) >> 16;
  hi = (hi + 0x7fffu + ((hi >> 16) & 1u)) & 0xffff0000u;
  out[i] = lo | hi;
}
*reinterpret_cast<uint4*>(reinterpret_cast<unsigned short*>(y) + (size_t)t * H + d) =
    make_uint4(out[0], out[1], out[2], out[3]);
"""


@functools.cache
def _weighted_sum_hip_kernel():
    return mx.fast.cuda_kernel(
        name="omlx_inkling2_weighted_sum_hip",
        input_names=["down", "inverse", "weights"],
        output_names=["y"],
        source=_WEIGHTED_SUM_HIP,
    )


def weighted_sum(down: mx.array, inverse: mx.array, weights: mx.array) -> mx.array:
    if not _rocm():
        return minimax.weighted_sum(down, inverse, weights)
    tokens, each = weights.shape
    hidden = down.shape[1]
    return _weighted_sum_hip_kernel()(
        inputs=[down, inverse.astype(mx.uint32), weights],
        template=[("TOPK", each), ("H", hidden)],
        grid=(tokens * (hidden // 8), 1, 1),
        threadgroup=(hidden // 8, 1, 1),
        output_shapes=[(tokens, hidden)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _relative(got: mx.array, want: mx.array) -> float:
    a, b = got.astype(mx.float32), want.astype(mx.float32)
    return (mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-30)).item()


def _guarded(name: str, check) -> None:
    """Run one kernel's self-check; on ROCm a CUDA-only kernel is skipped and
    a kernel that fails to build stays off."""
    if _rocm() and name.split(":")[0] in _CUDA_ONLY:
        return
    try:
        check()
    except Exception as error:  # noqa: BLE001 - a build or launch failure
        _on.discard(name)
        logger.warning("Inkling 2 %s kernel failed (%s); using plain ops", name, error)


def _check(name: str, ok: bool, detail: str) -> None:
    if ok:
        _on.add(name)
    else:
        _on.discard(name)
        logger.warning(
            "Inkling 2 %s kernel differs (%s); using plain ops", name, detail
        )


def enable(cfg, dtype: mx.Dtype) -> None:
    """Self-check each kernel against the plain ops of ``layers`` on small
    inputs of this model's shapes and keep the ones that agree."""
    if not available() or dtype != mx.bfloat16:
        return
    _on.clear()
    mx.random.seed(0)

    def check_attention():
        # A sliding and a full layer, queries past the first key tile.
        heads, kv, dim = cfg.swa_heads, cfg.swa_kv_heads, cfg.swa_head_dim
        worst = 0.0
        for window, extent, length, total in (
            (cfg.window, cfg.window, 150, 150 + cfg.window),
            (0, cfg.rel_extent, 150, 400),
        ):
            q = (mx.random.normal((1, length, heads, dim)) * 0.15).astype(dtype)
            k = (mx.random.normal((1, kv, total, dim)) * 0.15).astype(dtype)
            v = mx.random.normal((1, kv, total, dim)).astype(dtype)
            rel = (mx.random.normal((1, length, heads, extent)) * 0.5).astype(dtype)
            sink = mx.random.normal((1, length, heads)).astype(dtype)
            got = attention(q, k, v, rel, sink, dim**-0.5, window)
            want = layers.masked_attention(
                q.transpose(0, 2, 1, 3), k, v, rel, sink, dim**-0.5, window
            ).transpose(0, 2, 1, 3)
            worst = max(worst, _relative(got, want))
        _check("attention", worst <= 1e-2, f"relative error {worst:.2e}")

    def check_mix():
        # A full layer's three queries over every block.
        blocks = [
            layers.unit_rms(mx.random.normal((1, 5, cfg.hidden)))
            for _ in range(MIX_SOURCES - 1)
        ]
        partial = layers.unit_rms(mx.random.normal((1, 5, cfg.hidden)))
        query = mx.random.normal((3, cfg.hidden)) * 0.05
        scale = cfg.hidden**-0.5
        src = mx.stack([*blocks, partial], axis=-2)
        weights = mx.softmax((src @ query.T) * scale, axis=-2, precise=True)
        error = _relative(
            mix(blocks, partial, query, scale), weights.swapaxes(-1, -2) @ src
        )
        _check("mix", error <= 1e-5, f"relative error {error:.2e}")

    def check_route():
        # The same experts in the same order and the same weights.
        logits = mx.random.normal((64, cfg.experts)) * 2
        bias = mx.random.normal((cfg.experts,)) * 0.05
        inds, w = route_topk(logits, bias, cfg.top_k, cfg.route_scale)
        scores = mx.sigmoid(logits)
        order = mx.argsort(-(scores + bias), axis=-1)[:, : cfg.top_k]
        want_w = mx.take_along_axis(scores, order, axis=-1)
        want_w = want_w / (mx.sum(want_w, axis=-1, keepdims=True) + 1e-20) * cfg.route_scale
        same = mx.array_equal(inds, order.astype(mx.uint32)).item()
        error = _relative(w, want_w)
        _check("route", same and error <= 1e-6, f"same={same}, error {error:.2e}")

    def check_router_logits():
        # The MiniMax kernel's sums, a few rows per block.
        x = mx.random.normal((70, cfg.hidden)).astype(dtype)
        columns = mx.random.normal((cfg.hidden, cfg.experts)) * 0.02
        same = mx.array_equal(router_logits(x, columns), minimax.router(x, columns))
        _check("router_logits", same.item(), "not equal to the MiniMax kernel")

    def check_conv():
        same = True
        for channels in (cfg.hidden, cfg.kv_heads * cfg.head_dim):
            x = mx.random.normal((1, 37, channels)).astype(dtype)
            w = (mx.random.normal((channels, cfg.conv_k)) * 0.3).astype(dtype)
            state = mx.random.normal((1, cfg.conv_k - 1, channels))
            got = static_conv(x, w, state, 0.4472)
            want = layers.plain_static_conv(x, w, state, 0.4472)
            same = same and all(mx.array_equal(a, b).item() for a, b in zip(got, want))
        _check("conv", same, "not equal to the plain ops")

    def check_activation():
        experts = layers.Experts.__new__(layers.Experts)
        experts.caps = (
            mx.array(cfg.gate_cap, dtype=mx.float32),
            mx.array(cfg.up_cap, dtype=mx.float32),
        )
        experts.w13 = type("Bank", (), {})()
        experts.w13.post = mx.random.uniform(0.001, 0.02, (cfg.experts,))
        y = (mx.random.normal((70, 2 * cfg.inter)) * 3).astype(dtype)
        picked = mx.random.randint(0, cfg.experts, (70,)).astype(mx.uint32)
        got = activation(y, picked, experts.w13.post, experts.caps)
        want = experts.plain_activation(y, picked)
        _check("activation", mx.array_equal(got, want).item(), "not equal to plain ops")

    def check_experts(spec):
        # Each carrier against MLX's gather_qmm, which multiplies with the
        # weights the host's does (mx.dequantize of 3-bit codes differs).
        rows = mx.array([0] * 70 + [1] * 60, dtype=mx.uint32)
        x = mx.random.normal((130, 256)).astype(dtype)
        # BF16 weights, so affine scales and biases are BF16 as checkpoints hold them.
        w = (mx.random.normal((2, 128, 256)) * 0.05).astype(dtype)
        weight, scales, *biases = mx.quantize(w, **spec)
        named = {"b.weight": weight, "b.scales": scales}
        if biases:
            named["b.biases"] = biases[0]
        quant = layers.expert_spec(spec if spec["mode"] == "affine" else spec["mode"])
        bank = layers.Experts.Bank(named, "b", quant)
        want = bank(x[:, None], rows)[:, 0]
        error = _relative(gather(x, bank, rows), want)
        _check(f"experts:{_format(bank)}", error <= 1e-3, f"relative error {error:.2e}")

    def check_weighted_sum():
        # The weighted sum of each token's expert rows, a thread per 8 columns.
        down = mx.random.normal((70 * cfg.top_k, cfg.hidden)).astype(dtype)
        inverse = mx.random.permutation(70 * cfg.top_k).astype(mx.uint32)
        weights = mx.random.uniform(0.1, 2.0, (70, cfg.top_k))
        want = down[inverse].reshape(70, cfg.top_k, -1).astype(mx.float32)
        want = mx.sum(want * weights[..., None], axis=-2).astype(dtype)
        error = _relative(weighted_sum(down, inverse, weights), want)
        _check("weighted_sum", error <= 1e-3, f"relative error {error:.2e}")

    def check_nvfp4():
        # Two experts, rows of each: the ROCm NVFP4 kernel against gather_qmm.
        w = (mx.random.normal((2, 128, 256)) * 0.05).astype(dtype)
        weight, scales = mx.quantize(w, mode="nvfp4")
        bank = layers.Experts.Bank(
            {"b.weight": weight, "b.scales": scales}, "b", layers.expert_spec("nvfp4")
        )
        rows = mx.array([0] * 40 + [1] * 30, dtype=mx.uint32)
        x = mx.random.normal((70, 256)).astype(dtype)
        want = bank(x[:, None], rows)[:, 0]
        error = _relative(nvfp4_gemm(x, nvfp4_pack(bank), rows), want)
        _check("nvfp4", error <= 1e-3, f"relative error {error:.2e}")

    def check_clip():
        # A full layer's q and r slices of one projection, with tau.
        heads, dim = cfg.heads, cfg.head_dim
        width = heads * dim + heads * cfg.d_rel + 40
        base = (mx.random.normal((1, 70, width)) * 3).astype(dtype)
        scale = mx.random.uniform(0.5, 2.0, (heads, dim))
        tau = mx.random.uniform(1.0, 1.3, (1, 70, 1, 1))
        worst = 0.0
        for offset, d, sc, threshold, t in (
            (0, dim, scale, cfg.qk_clip, tau),
            (heads * dim, cfg.d_rel, None, cfg.rel_clip, None),
        ):
            x = base[..., offset : offset + heads * d].reshape(1, 70, heads, d)
            want = layers.l2_clip(x, sc, threshold)
            if t is not None:
                want = (want.astype(mx.float32) * t).astype(want.dtype)
            got = l2_clip(base, offset, heads, d, sc, threshold, t)
            worst = max(worst, _relative(got, want))
        _check("clip", worst <= 1e-2, f"relative error {worst:.2e}")

    def check_gate():
        width = cfg.heads * cfg.head_dim
        attn = mx.random.normal((1, 70, width)).astype(dtype)
        base = mx.random.normal((1, 70, width + 72)).astype(dtype)
        g = base[..., 40 : 40 + width]
        want = (attn.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(dtype)
        error = _relative(gate(attn, base, 40), want)
        _check("gate", error <= 1e-2, f"relative error {error:.2e}")

    def check_messages():
        # Three inputs from a full layer's mix, then one from query 0.
        hidden, rank = cfg.hidden, 64
        mixed = mx.random.normal((1, 70, 3, hidden)) * 0.7
        messages = mx.random.normal((1, 70, 4, hidden))
        down = (mx.random.normal((4, 2, hidden, rank)) * 0.02).astype(dtype)
        gf = (mx.random.normal((4, rank, 2 * hidden)) * 0.05).astype(dtype).astype(mx.float32)
        weight = mx.random.uniform(0.5, 1.5, (hidden,)).astype(dtype)
        worst = 0.0
        for m0, n, mx_in in ((0, 3, mixed), (3, 1, mixed[:, :, :1])):
            x = mx_in.transpose(2, 0, 1, 3) if mx_in.shape[2] == n else mx_in[:, :, :1].transpose(2, 0, 1, 3)
            msg = messages[:, :, m0 : m0 + n].transpose(2, 0, 1, 3)
            d = down[m0 : m0 + n]
            want = layers.mix_messages(x, msg, d.astype(mx.float32), gf[m0 : m0 + n])
            want = mx.fast.rms_norm(want, weight, cfg.eps).astype(dtype)
            got = mix_messages_normed(
                mx_in, messages, m0, n, d.reshape(n, 2 * hidden, rank),
                gf[m0 : m0 + n], weight, cfg.eps,
            )
            worst = max(worst, _relative(got, want))
        _check("messages", worst <= 1e-2, f"relative error {worst:.2e}")

    def check_dispatch():
        inds = mx.random.randint(0, cfg.experts, (300, cfg.top_k)).astype(mx.uint32)
        weights = mx.random.uniform(0.1, 1.0, (300, cfg.top_k))
        post = mx.random.uniform(0.5, 2.0, (cfg.experts,))
        offsets, first, source, sorted_, inverse, scaled = dispatch(inds, weights, cfg.experts, post)
        flat = inds.reshape(-1)
        counts = mx.zeros((cfg.experts,), dtype=mx.int32).at[flat.astype(mx.int32)].add(1)
        tile = nvfp4_tile_rows()
        ok = mx.array_equal(offsets[1:], mx.cumsum(counts)).item()
        ok = ok and mx.array_equal(first[1:], mx.cumsum((counts + tile - 1) // tile)).item()
        ok = ok and mx.array_equal(sorted_[inverse], flat).item()
        tokens = mx.arange(300 * cfg.top_k) // cfg.top_k
        ok = ok and mx.array_equal(source[inverse], tokens).item()
        ok = ok and mx.array_equal(mx.sort(inverse), mx.arange(300 * cfg.top_k)).item()
        ok = ok and mx.allclose(scaled, weights * post[inds]).item()
        _check("dispatch", ok, "the routed rows do not match their entries")

    if _rocm():
        _guarded("nvfp4", check_nvfp4)
        _guarded("dispatch", check_dispatch)
        _guarded("messages", check_messages)
        _guarded("clip", check_clip)
        _guarded("gate", check_gate)
    _guarded("attention", check_attention)
    _guarded("mix", check_mix)
    _guarded("route", check_route)
    _guarded("router_logits", check_router_logits)
    _guarded("conv", check_conv)
    _guarded("activation", check_activation)
    carriers = [{"mode": m} for m in _FORMATS]
    carriers += [{"mode": "affine", "bits": b, "group_size": 64} for b in _AFFINE_BITS]
    for spec in carriers:
        name = spec["mode"] if spec["mode"] != "affine" else f"affine{spec['bits']}"
        _guarded(f"experts:{name}", lambda spec=spec: check_experts(spec))
    if cfg.hidden <= 8192 and cfg.hidden % 8 == 0:
        _guarded("weighted_sum", check_weighted_sum)
