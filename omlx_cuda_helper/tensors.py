# SPDX-License-Identifier: Apache-2.0
"""MLX-side helpers shared by the family stages (workers only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from . import checkpoint, oqh
from .wire import HostArray

# Cache rows grow every chunk, so a freed buffer never fits again; cap what MLX keeps.
_CACHE_BYTES = 1 << 30


def limit_memory(limit: int) -> None:
    """Hold MLX's buffers, cached ones included, under *limit* bytes: past it
    the allocator frees its cache before it allocates."""
    mx.set_cache_limit(_CACHE_BYTES)
    if limit:
        mx.set_memory_limit(limit)


def release_cache() -> None:
    mx.clear_cache()


def peak_bytes() -> int:
    """Peak GPU bytes since the last call."""
    peak = mx.get_peak_memory()
    mx.reset_peak_memory()
    return peak


def check_backend() -> None:
    """Refuse an MLX CUDA build without the patches in mlx-build/.

    Stock MLX rounds twice when it rebuilds an affine weight in a 16-bit type
    and lets cuBLAS round the partial sums of some 16-bit products.
    """
    if not mx.cuda.is_available():
        return
    mx.random.seed(0)
    x = mx.random.normal((1, 8192, 2560)).astype(mx.bfloat16)
    w = (mx.random.normal((2560, 512)) * 0.02).astype(mx.bfloat16)
    once = (x.astype(mx.float32) @ w.astype(mx.float32)).astype(mx.bfloat16)
    equal = (x @ w == once).astype(mx.float32).mean().item()
    packed, scales, biases = mx.quantize(
        mx.random.normal((64, 256)) * 0.02, group_size=64, bits=4
    )
    low = [scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)]
    wide = [a.astype(mx.float32) for a in low]
    rebuilt = mx.dequantize(packed, *low, group_size=64, bits=4)
    exact = mx.dequantize(packed, *wide, group_size=64, bits=4).astype(mx.bfloat16)
    if equal < 0.99 or not mx.array_equal(rebuilt, exact).item():
        raise RuntimeError(
            "this MLX CUDA build rounds 16-bit products or affine weights "
            "more than once; build MLX with the patches in mlx-build/"
        )


def to_host(array: mx.array) -> HostArray:
    """An evaluated array as contiguous numpy data and its MLX dtype name."""
    dtype = str(array.dtype).rsplit(".", 1)[-1]
    if array.dtype == mx.bfloat16:
        # numpy has no bfloat16; the bits travel as uint16.
        array = array.view(mx.uint16)
    array = mx.contiguous(array)
    mx.eval(array)
    return np.ascontiguousarray(np.asarray(array)), dtype


def from_host(value: HostArray) -> mx.array:
    data, dtype = value
    array = mx.array(data)
    if dtype == "bfloat16":
        array = array.view(mx.bfloat16)
    return array


# Set by the worker where kernels read CUDA managed memory slowly (an integrated
# GPU, WSL): weights then load on the GPU stream, into the GPU's memory pool.
weights_on_gpu = False

# Set by the worker when a family keeps its affine weights packed to hold more
# layers.
packed_weights = False


def load(model_dir: str | Path, names: list[str]) -> dict[str, mx.array]:
    """The named tensors, lazy until evaluated. Each shard is opened once."""
    root = Path(model_dir)
    index = checkpoint.tensors(root)
    shards: dict[str, dict[str, mx.array]] = {}
    out = {}
    for name in names:
        file = index[name].file
        if file not in shards:
            stream = mx.gpu if weights_on_gpu else None
            shards[file] = mx.load(str(root / file), stream=stream)
        out[name] = shards[file][name]
    return out


def unpack(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    quant: dict[str, Any],
    dtype: mx.Dtype | None = None,
) -> mx.array:
    """The dense weight of a quantized tensor, rounded once.

    An affine weight is rebuilt in float32 and rounded to the target type
    afterwards, which gives the values Metal's quantized matmul multiplies with.
    """
    mode = quant.get("mode", "affine")
    sizes = {"group_size": quant["group_size"], "bits": quant["bits"]}
    if mode == "affine":
        target = dtype or scales.dtype
        if biases is not None:
            biases = biases.astype(mx.float32)
        dense = mx.dequantize(weight, scales.astype(mx.float32), biases, **sizes)
        return dense.astype(target)
    # The floating point modes store exponents, not scales of a float type.
    return mx.dequantize(weight, scales, mode=mode, dtype=dtype or mx.bfloat16, **sizes)


class Linear:
    """A bias-free linear layer from a dense or a quantized weight.

    ``dense=True`` unpacks a quantized weight at load: the dense product is
    about twice as fast on CUDA and costs the unpacked bytes of GPU memory.
    """

    def __init__(
        self,
        weights: dict[str, mx.array],
        prefix: str,
        quant: dict[str, Any] | None,
        *,
        dense: bool = True,
        dtype: mx.Dtype | None = None,
    ):
        weight = weights[f"{prefix}.weight"]
        self.quant = None
        # oQh: the input rotation (signs, block) and the row scale of wide codes.
        self.rotation = None
        self.post = None
        if oqh.is_oqh(weights, prefix):
            if not dense:
                raise ValueError(f"oQh {prefix} is read only as a dense weight")
            scales, biases, sizes, self.post = oqh.affine_view(weights, prefix)
            signs = weights[f"{prefix}.{oqh.SIGNS}"]
            self.rotation = (signs, int(signs.shape[-1]))
            weight = unpack(weight, scales, biases, sizes, dtype or scales.dtype)
        elif f"{prefix}.scales" in weights:
            scales, biases = weights[f"{prefix}.scales"], weights.get(
                f"{prefix}.biases"
            )
            if dense:
                weight = unpack(weight, scales, biases, quant, dtype)
            else:
                self.quant = (scales, biases, quant)
        elif dtype is not None:
            weight = weight.astype(dtype)
        self.weight = weight
        # True once ``weight`` is held as [in, out].
        self.columns = False

    @property
    def dtype(self) -> mx.Dtype:
        """The dtype the layer computes in: the scales' for a packed weight."""
        return self.weight.dtype if self.quant is None else self.quant[0].dtype

    def lay_columns(self) -> None:
        """Hold a dense weight as [in, out], contiguous.

        hipBLASLt multiplies by a stored operand faster than by a transposed
        view of one: 35.9 against 24.6 TFLOP/s for the 17408 to 5120
        projection on a Radeon 8060S.
        """
        if self.quant is None and not self.columns:
            self.weight = mx.contiguous(self.weight.T)
            self.columns = True

    def _product(self, x: mx.array) -> mx.array:
        return x @ (self.weight if self.columns else self.weight.T)

    def parameters(self) -> list[mx.array]:
        extra = [a for a in self.quant[:2] if a is not None] if self.quant else []
        if self.rotation is not None:
            extra.append(self.rotation[0])
        if self.post is not None:
            extra.append(self.post)
        return [self.weight, *extra]

    def rotated(self, x: mx.array) -> mx.array | None:
        """x in this layer's oQh basis, None for a layer without one."""
        return None if self.rotation is None else oqh.rotate(x, *self.rotation)

    def rows(self, x: mx.array, start: int, end: int | None) -> mx.array:
        """Output rows ``[start:end]`` of ``x @ W.T``, as the host projects a
        slice of a Linear: through those rows of the weight only."""
        if self.rotation is not None:
            return self(x)[..., start:end]
        if self.quant is None and self.columns:
            # A column slice of the [in, out] weight is not contiguous, and a
            # matmul would copy it each call: keep each slice once.
            held = self.__dict__.setdefault("_slices", {})
            if (start, end) not in held:
                held[(start, end)] = mx.contiguous(self.weight[:, start:end])
            return x @ held[(start, end)]
        if self.quant is None:
            return x @ self.weight[start:end].T
        scales, biases, quant = self.quant
        return mx.quantized_matmul(
            x,
            self.weight[start:end],
            scales[start:end],
            None if biases is None else biases[start:end],
            transpose=True,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
        )

    def __call__(self, x: mx.array, rotated: mx.array | None = None) -> mx.array:
        """x @ W.T; an oQh layer takes ``rotated`` (its basis) or rotates x."""
        if self.rotation is not None:
            x = self.rotated(x) if rotated is None else rotated
            y = self._product(x)
            # The host rounds the product, then the row scale product.
            return y if self.post is None else y * self.post
        if self.quant is None:
            return self._product(x)
        scales, biases, quant = self.quant
        return mx.quantized_matmul(
            x,
            self.weight,
            scales,
            biases,
            transpose=True,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
        )
