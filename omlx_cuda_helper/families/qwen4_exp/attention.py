# SPDX-License-Identifier: Apache-2.0
"""Qwen sparse attention (QSA) for prefill, batch one and text only.

A query that sees more than ``top_k`` complete blocks of ``ratio`` keys attends
to its ``top_k`` best blocks and the incomplete tail, others to every key.

The rotary embedding and the selection mask are ports from mlx-vlm (MIT,
Copyright (c) 2025 Prince Canuma), the gathered arm from oMLX.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any

import mlx.core as mx
import numpy as np

from . import hip, kernels
from .ops import Weights, norm, unit_scale

# Chunks shorter than this stay on masked dense attention on the host.
GATHERED_MIN_QUERIES = 16
# Query rows of one gathered reference step; bounds the gathered key copy.
_REFERENCE_ROWS = 128


class QSACache:
    """Keys, values and indexer state of one QSA layer for one prompt."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.index_keys: mx.array | None = None
        self.positions: mx.array | None = None
        # Normalized, rotated mean key of each complete block.
        self.pooled: mx.array | None = None
        # Values of the complete blocks as the HIP attention kernel reads them.
        self.blocked: mx.array | None = None
        self.offset = 0
        # (keys, values, index keys, positions) of the last chunk.
        self.appended: tuple[mx.array, ...] = ()

    def append(
        self,
        keys: mx.array,
        values: mx.array,
        index_keys: mx.array,
        positions: mx.array,
    ) -> None:
        """Add rows: keys/values [1, heads, n, dim], index keys [1, n, dim]."""
        self.appended = (keys, values, index_keys, positions)
        if self.keys is None:
            self.keys, self.values = keys, values
            self.index_keys, self.positions = index_keys, positions
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            self.index_keys = mx.concatenate([self.index_keys, index_keys], axis=1)
            self.positions = mx.concatenate([self.positions, positions], axis=1)
        self.offset = self.keys.shape[2]

    def arrays(self) -> list[mx.array]:
        held = (self.keys, self.values, self.index_keys, self.positions)
        held += (self.pooled, self.blocked)
        return [a for a in held if a is not None]


@functools.cache
def _head_sum(heads: int, dim: int):
    """Compiled ``[rows, heads, blocks] -> [rows, blocks]`` block scores."""

    def total(scores: mx.array) -> mx.array:
        out = mx.maximum(scores[:, 0], 0)
        for head in range(1, heads):
            out = out + mx.maximum(scores[:, head], 0)
        return out / math.sqrt(dim)

    return mx.compile(total)


class Attention:
    """Gated multi-head attention over the QSA selection of cached tokens."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim") or text["hidden_size"] // self.heads
        self.eps = text["rms_norm_eps"]
        self.q_proj = w.linear("q_proj")
        self.k_proj = w.linear("k_proj")
        self.v_proj = w.linear("v_proj")
        self.o_proj = w.linear("o_proj")
        self.q_scale = unit_scale(w["q_norm.weight"])
        self.k_scale = unit_scale(w["k_norm.weight"])

        self.index_heads = text.get("indexer_n_heads", 4)
        self.index_dim = text.get("indexer_head_dim", 128)
        self.ratio = text.get("indexer_compress_ratio", 4)
        # STRIX_QPF2K_QSA_BUDGET=1536 is the fast profile. On an 8,192-row
        # forward it lengthens the sparse tail and the full model got slower,
        # so the checkpoint budget stays unless that variable is set.
        checkpoint = int(text.get("indexer_budget", 2048))
        requested = os.environ.get("STRIX_QPF2K_QSA_BUDGET", "")
        self.budget = int(requested) if requested[:1] not in ("", "0") else checkpoint
        if self.budget % self.ratio:
            raise ValueError(f"indexer budget {self.budget} is not a multiple of {self.ratio}")
        self.top_k = self.budget // self.ratio
        self.index_proj = w.linear("indexer.index_qk_proj")
        self.index_q_scale = unit_scale(w["indexer.q_layernorm.weight"])
        self.index_k_scale = unit_scale(w["indexer.k_layernorm.weight"])

        rope = text.get("rope_parameters") or {}
        rotary = int(self.head_dim * rope.get("partial_rotary_factor", 0.25))
        if rotary > min(self.head_dim, self.index_dim):
            raise ValueError("The rotary width exceeds an attention head")
        # Float32 on the CPU: every helper build rotates by the same angles.
        exponent = np.arange(0, rotary, 2, dtype=np.float32) / np.float32(rotary)
        base = np.float32(rope.get("rope_theta", 10_000_000))
        self.inv_freq = mx.array((np.float32(1.0) / base**exponent).astype(np.float32))

        self._head_sum = _head_sum(self.index_heads, self.index_dim)
        self.kernel: kernels.SparseAttention | hip.SparseAttention | None = None
        self.selection: kernels.BlockSelection | hip.BlockSelection | None = None
        # The kernels module of the fused ``prepare``, once it passes.
        self.fused_prep: Any = None

    def enable_kernels(self) -> None:
        """Use the prep, selection and attention kernels that reproduce the ops."""
        if not kernels.available():
            self.enable_hip()
            return
        dtype = self.q_proj.weight.dtype
        geometry = (self.heads, self.kv_heads, self.head_dim, self.index_heads)
        half = self.inv_freq.shape[0]
        if kernels.qsa_prep_fits(*geometry, self.index_dim, half, dtype):
            self.fused_prep = kernels
            if not self._prepares_like(dtype):
                self.fused_prep = None
        if kernels.BlockSelection.fits(self.index_heads, self.index_dim, dtype):
            selection = kernels.BlockSelection(
                self.index_heads, self.index_dim, self.ratio, self.top_k
            )
            if self._selects_like(selection, dtype):
                self.selection = selection
        if not kernels.SparseAttention.fits(
            self.heads, self.kv_heads, self.head_dim, dtype
        ):
            return
        kernel = kernels.SparseAttention(
            self.heads, self.kv_heads, self.head_dim, self.ratio, self.top_k
        )
        if self._attends_like(kernel, dtype):
            self.kernel = kernel

    def enable_hip(self) -> None:
        """Use the HIP attention kernel if it reproduces the ops."""
        dtype = self.q_proj.weight.dtype
        geometry = (self.heads, self.kv_heads, self.head_dim, self.ratio)
        if not hip.available():
            return
        prep = (self.heads, self.kv_heads, self.head_dim, self.index_heads)
        half = self.inv_freq.shape[0]
        if self.fused_prep is None and hip.qsa_prep_fits(
            *prep, self.index_dim, half, dtype
        ):
            self.fused_prep = hip
            if not self._prepares_like(dtype):
                self.fused_prep = None
        if self.selection is None and hip.BlockSelection.fits(
            self.index_heads, self.index_dim, dtype
        ):
            selection = hip.BlockSelection(
                self.index_heads, self.index_dim, self.ratio, self.top_k
            )
            if self._picks_like(selection, dtype):
                self.selection = selection
        if self.kernel is not None or not hip.SparseAttention.fits(*geometry, dtype):
            return
        kernel = hip.SparseAttention(*geometry, self.top_k)
        if self._attends_like(kernel, dtype):
            self.kernel = kernel

    def _attends_like(self, kernel, dtype: mx.Dtype) -> bool:
        """Whether ``kernel`` gives what ``_gathered`` gives."""
        # Rows around the point where a query stops taking every block, so
        # short selections, every tail length and full selections all occur.
        total = self.budget + 16 * self.ratio
        rows = min(24 * self.ratio, total - self.ratio)
        shape = (1, self.kv_heads, total, self.head_dim)
        keys = mx.random.normal(shape).astype(dtype)
        values = mx.random.normal(shape).astype(dtype)
        queries = mx.random.normal((1, self.heads, rows, self.head_dim)).astype(dtype)
        visible = total - rows + mx.arange(rows, dtype=mx.int32) + 1
        complete = visible // self.ratio
        blocks = total // self.ratio
        # Only complete blocks, as the indexer does in production. Random
        # picks that reach past ``complete`` are not a real selection and
        # disagree with the shared-key kernel's filling-block mask.
        scores = mx.random.uniform(shape=(rows, blocks))
        scores = mx.where(
            mx.arange(blocks)[None, :] < complete[:, None],
            scores,
            mx.zeros_like(scores) - 1,
        )
        picks = mx.sort(
            mx.argsort(scores, axis=-1)[:, -self.top_k :].astype(mx.int32),
            axis=-1,
        )
        first = mx.arange(self.top_k, dtype=mx.int32)[None]
        selected = mx.where((complete <= self.top_k)[:, None], first, picks)
        want = self._gathered(queries, keys, values, selected, visible)
        got = kernel(queries, keys, values, selected, total - rows + 1)
        return kernels.close(got, want, "QSA attention")

    def _picks_like(self, selection: hip.BlockSelection, dtype: mx.Dtype) -> bool:
        """Whether ``selection`` picks blocks that score what ``_select``'s do.

        The float32 scores of two builds differ in their last bits, so a pick
        may differ where two blocks all but tie: each row's picks are distinct
        visible blocks in order, and their reference scores are the best ones.
        """
        ratio, blocks = self.ratio, self.top_k + 96
        # Rows on both sides of ``top_k`` complete blocks, none past the bank.
        rows = 100 * ratio
        bank = mx.random.normal((1, blocks, self.index_dim)).astype(dtype)
        queries = mx.random.normal((rows, self.index_heads, self.index_dim))
        queries = queries.astype(dtype)
        first = (self.top_k - 8) * ratio
        got = selection(queries, bank, first)
        want = self._select(queries, bank, first)
        complete = (first + mx.arange(rows, dtype=mx.int32)) // ratio
        scores = self._block_scores(queries, bank)
        ordered = mx.all(got[:, 1:] > got[:, :-1]) & mx.all(got[:, 0] >= 0)
        visible = mx.all(got[:, -1] < mx.maximum(complete, self.top_k))
        mine = mx.sort(mx.take_along_axis(scores, got, axis=-1), axis=-1)
        best = mx.sort(mx.take_along_axis(scores, want, axis=-1), axis=-1)
        same = mx.allclose(mine, best, rtol=1e-5, atol=1e-6)
        if ordered.item() and visible.item() and same.item():
            return True
        kernels.logger.warning("QSA selection kernel differs; using the reference")
        return False

    def _prepares_like(self, dtype: mx.Dtype) -> bool:
        """Whether ``_prepare_fused`` gives what ``prepare`` gives."""
        rows, past = 64, 1000
        widths = (
            self.q_proj.shape[0],
            self.k_proj.shape[0],
            self.v_proj.shape[0],
            self.index_proj.shape[0],
        )
        outs = [mx.random.normal((1, rows, w)).astype(dtype) for w in widths]
        positions = mx.arange(past, past + rows, dtype=mx.int32)
        want = self.prepare(*outs, positions)
        got = self._prepare_fused(*outs, past)
        names = ("queries", "gate", "keys", "values", "index keys", "index queries")
        return all(
            kernels.close(a, b, f"QSA {name}") for a, b, name in zip(got, want, names)
        )

    def _prepare_fused(
        self,
        q_out: mx.array,
        k_out: mx.array,
        v_out: mx.array,
        index: mx.array,
        past: int,
    ) -> tuple[mx.array, ...]:
        """``prepare`` in one launch for rows at positions ``past`` on."""
        rows = q_out.shape[1]
        gate = q_out.reshape(1, rows, self.heads, 2, -1)[..., 1, :].reshape(1, rows, -1)
        queries, keys, values, index_keys, index_queries = self.fused_prep.qsa_prep(
            self, q_out, k_out, v_out, index, past
        )
        return queries, gate, keys, values, index_keys, index_queries

    def _selects_like(self, selection: kernels.BlockSelection, dtype) -> bool:
        """Whether ``selection`` picks exactly what ``_select`` picks.

        Small integer inputs give sums that do not depend on their order and
        many tied scores, and rows on both sides of ``top_k`` complete blocks.
        """
        ratio, blocks = self.ratio, self.top_k + 64
        rows = 72 * ratio
        bank = mx.random.randint(-1, 2, (1, blocks, self.index_dim)).astype(dtype)
        shape = (rows, self.index_heads, self.index_dim)
        queries = mx.random.randint(-1, 2, shape).astype(dtype)
        first = (self.top_k - 8) * ratio
        got = selection(queries, bank, first)
        if mx.array_equal(got, self._select(queries, bank, first)).item():
            return True
        kernels.logger.warning("QSA selection kernel differs; using the reference")
        return False

    def parameters(self) -> list[mx.array]:
        linears = (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.index_proj)
        scales = [self.q_scale, self.k_scale, self.index_q_scale, self.index_k_scale]
        return scales + [a for layer in linears for a in layer.parameters()]

    def _rope(self, x: mx.array, positions: mx.array) -> mx.array:
        """Rotate the leading dims of [1, heads, n, dim] by text positions [n]."""
        half = self.inv_freq.shape[0]
        angle = positions.astype(mx.float32)[:, None] * self.inv_freq
        cos, sin = mx.cos(angle), mx.sin(angle)
        a = x[..., :half].astype(mx.float32)
        b = x[..., half : 2 * half].astype(mx.float32)
        parts = [
            (a * cos - b * sin).astype(x.dtype),
            (b * cos + a * sin).astype(x.dtype),
        ]
        if x.shape[-1] > 2 * half:
            parts.append(x[..., 2 * half :])
        return mx.concatenate(parts, axis=-1)

    def _pooled(self, cache: QSACache) -> mx.array:
        """The block bank [1, blocks, dim], extended by the new complete blocks."""
        ratio = self.ratio
        have = 0 if cache.pooled is None else cache.pooled.shape[1]
        complete = cache.offset // ratio
        if have < complete:
            raw = cache.index_keys[:, have * ratio : complete * ratio]
            raw = raw.reshape(1, complete - have, ratio, self.index_dim)
            pooled = mx.mean(raw.astype(mx.float32), axis=-2).astype(raw.dtype)
            pooled = norm(pooled, self.index_k_scale, self.eps)
            # A block is rotated by the position of its first token.
            starts = mx.squeeze(cache.positions, 0)[have * ratio : complete * ratio : ratio]
            pooled = self._rope(pooled[:, None], starts)[:, 0]
            if cache.pooled is not None:
                pooled = mx.concatenate([cache.pooled, pooled], axis=1)
            cache.pooled = pooled
        return cache.pooled

    def _block_scores(self, index_queries: mx.array, pooled: mx.array) -> mx.array:
        """Float32 score [rows, blocks] of each block for each query.

        Float32 products, heads summed in host order; rounding would change the picks.
        """
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        scores = index_queries.astype(mx.float32).reshape(
            rows * self.index_heads, self.index_dim
        ) @ mx.squeeze(pooled, 0).astype(mx.float32).swapaxes(-1, -2)
        return self._head_sum(scores.reshape(rows, self.index_heads, blocks))

    def _select(
        self, index_queries: mx.array, pooled: mx.array, first_visible: int
    ) -> mx.array:
        """Each query's selected blocks in key order, [rows, width] int32.

        ``index_queries`` [rows, heads, dim] are normalized and rotated and
        row ``i`` sees ``first_visible + i`` tokens. A query with at most
        ``top_k`` complete blocks gets the first ``width`` block ids.
        """
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        if blocks <= self.top_k:
            first = mx.arange(blocks, dtype=mx.int32)[None]
            return mx.broadcast_to(first, (rows, blocks))
        if self.selection is not None:
            return self.selection(index_queries, pooled, first_visible)
        scores = self._block_scores(index_queries, pooled)
        complete = (first_visible + mx.arange(rows, dtype=mx.int32)) // self.ratio
        scores = mx.where(mx.arange(blocks)[None] < complete[:, None], scores, -mx.inf)
        ranked = mx.argpartition(scores, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        first = mx.arange(self.top_k, dtype=mx.int32)[None]
        ranked = mx.where((complete <= self.top_k)[:, None], first, ranked)
        return mx.sort(ranked.astype(mx.int32), axis=-1)

    def _selection_mask(
        self, selected: mx.array, visible: mx.array, keys: int
    ) -> mx.array:
        """Boolean attention mask [rows, keys] of a block selection."""
        rows, blocks = selected.shape[0], keys // self.ratio
        hits = mx.put_along_axis(
            mx.zeros((rows, blocks), dtype=mx.bool_),
            selected,
            mx.array(True),
            axis=-1,
        )
        tokens = mx.repeat(hits, self.ratio, axis=-1)
        if blocks * self.ratio < keys:
            pad = mx.zeros((rows, keys - blocks * self.ratio), dtype=mx.bool_)
            tokens = mx.concatenate([tokens, pad], axis=-1)
        complete = visible // self.ratio
        token = mx.arange(keys)[None]
        causal = token < visible[:, None]
        tail = (token >= (complete * self.ratio)[:, None]) & causal
        sparse = (complete > self.top_k)[:, None]
        return mx.where(sparse, tokens | tail, causal)

    def _gathered(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        selected: mx.array,
        visible: mx.array,
    ) -> mx.array:
        """Attention of [1, heads, rows, dim] queries over their selected keys.

        Returns [1, rows, heads, dim]. The scores and the softmax are float32.
        """
        ratio, rows, width = self.ratio, queries.shape[2], selected.shape[1]
        groups = self.heads // self.kv_heads
        complete = visible // ratio
        count = mx.minimum(complete, self.top_k)
        tokens = selected[..., None] * ratio + mx.arange(ratio, dtype=mx.int32)
        valid = mx.repeat(mx.arange(width)[None] < count[:, None], ratio, axis=-1)
        tail = complete[:, None] * ratio + mx.arange(ratio - 1, dtype=mx.int32)
        tokens = mx.concatenate([tokens.reshape(rows, width * ratio), tail], axis=-1)
        valid = mx.concatenate([valid, tail < visible[:, None]], axis=-1)
        flat = mx.where(valid, tokens, 0).reshape(-1)

        def gather(cached: mx.array) -> mx.array:
            picked = mx.squeeze(cached, 0).transpose(1, 0, 2)[flat]
            return picked.reshape(rows, -1, self.kv_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )

        grouped = (
            mx.squeeze(queries, 0)
            .transpose(1, 0, 2)
            .reshape(rows, self.kv_heads, groups, self.head_dim)
        )
        scores = (
            grouped.astype(mx.float32)
            @ gather(keys).astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(self.head_dim)
        scores = mx.where(valid[:, None, None, :], scores, -mx.inf)
        out = mx.softmax(scores, axis=-1) @ gather(values).astype(mx.float32)
        return out.astype(queries.dtype).reshape(1, rows, self.heads, self.head_dim)

    def _masked(
        self, queries: mx.array, keys: mx.array, values: mx.array, mask: mx.array
    ) -> mx.array:
        """Attention over the keys a boolean ``mask`` [rows, keys] allows.

        Returns [1, rows, heads, dim]. Scores and softmax are float32, like
        the gathered rows: a 16-bit score sheet moves the softmax weights.
        """
        rows = queries.shape[2]
        grouped = queries.reshape(1, self.kv_heads, -1, rows, self.head_dim)
        scores = (
            grouped.astype(mx.float32)
            @ keys[:, :, None].astype(mx.float32).swapaxes(-1, -2)
        ) / math.sqrt(self.head_dim)
        scores = mx.where(mask, scores, -mx.inf)
        out = mx.softmax(scores, axis=-1) @ values[:, :, None].astype(mx.float32)
        out = out.astype(queries.dtype).reshape(1, self.heads, rows, self.head_dim)
        return out.transpose(0, 2, 1, 3)

    def _sparse_rows(
        self,
        queries: mx.array,
        index_queries: mx.array,
        cache: QSACache,
        first: int,
    ) -> mx.array:
        """Gathered attention of the query rows from ``first`` to the end."""
        rows = queries.shape[2]
        pooled = self._pooled(cache)
        kernel = self.kernel if pooled.shape[1] > self.top_k else None
        step = kernel.rows_per_launch if kernel else _REFERENCE_ROWS
        packed = ()
        if isinstance(kernel, hip.SparseAttention):
            packed = (kernel.blocked(cache),)
        outputs = []
        for start in range(first, rows, step):
            stop = min(start + step, rows)
            seen = cache.offset - rows + start + 1
            visible = seen + mx.arange(stop - start, dtype=mx.int32)
            selected = self._select(index_queries[start:stop], pooled, seen)
            part = queries[:, :, start:stop]
            if kernel:
                out = kernel(part, cache.keys, cache.values, selected, seen, *packed)
            else:
                out = self._gathered(part, cache.keys, cache.values, selected, visible)
            outputs.append(out)
            # One step at a time: each holds a float32 sheet of its scores.
            mx.eval(out)
        return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)

    def prepare(
        self,
        q_out: mx.array,
        k_out: mx.array,
        v_out: mx.array,
        index: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, ...]:
        """Normalized, rotated operands from the four projections of a chunk.

        Returns (queries [1, heads, rows, dim], output gate [1, rows, -1],
        keys and values [1, kv heads, rows, dim], raw index keys
        [1, rows, index dim], index queries [rows, index heads, index dim]).
        """
        rows = q_out.shape[1]
        queries, gate = mx.split(q_out.reshape(1, rows, self.heads, -1), 2, axis=-1)
        queries = norm(queries, self.q_scale, self.eps).transpose(0, 2, 1, 3)
        keys = norm(k_out.reshape(1, rows, self.kv_heads, -1), self.k_scale, self.eps)
        keys = keys.transpose(0, 2, 1, 3)
        values = v_out.reshape(1, rows, self.kv_heads, -1).transpose(0, 2, 1, 3)
        index = index.reshape(rows, self.index_heads + 1, self.index_dim)
        index_queries = norm(index[:, : self.index_heads], self.index_q_scale, self.eps)
        # Rotated like the block bank.
        index_queries = self._rope(index_queries.swapaxes(0, 1)[None], positions)
        return (
            self._rope(queries, positions),
            gate.reshape(1, rows, -1),
            self._rope(keys, positions),
            values,
            index[None, :, self.index_heads],
            mx.squeeze(index_queries, 0).swapaxes(0, 1),
        )

    def __call__(self, x: mx.array, cache: QSACache) -> mx.array:
        _, rows, _ = x.shape
        past = cache.offset
        positions = mx.arange(past, past + rows, dtype=mx.int32)
        outs = (self.q_proj(x), self.k_proj(x), self.v_proj(x), self.index_proj(x))
        if self.fused_prep:
            prepared = self._prepare_fused(*outs, past)
        else:
            prepared = self.prepare(*outs, positions)
        queries, gate, keys, values, index_keys, index_queries = prepared
        cache.append(keys, values, index_keys, positions[None])

        total = past + rows
        token = mx.arange(total)[None]
        if rows >= GATHERED_MIN_QUERIES and total > self.budget:
            # Rows that see at most top_k complete blocks attend to all keys.
            # The causal kernel is asked about this prefix only: the rows past
            # it see more than top_k blocks, and asking about the whole tensor
            # makes the kernel refuse the prefix too.
            dense = min(rows, max(0, self.budget + self.ratio - 1 - past))
            if rows - dense == 1:
                dense -= 1
            outputs = []
            if dense:
                head = self._hip_causal(queries[:, :, :dense], cache, past + 1)
                if head is None:
                    allow = token[:, : past + dense] <= past + mx.arange(dense)[:, None]
                    head = self._masked(
                        queries[:, :, :dense],
                        cache.keys[:, :, : past + dense],
                        cache.values[:, :, : past + dense],
                        allow,
                    )
                outputs.append(head)
            if dense < rows:
                outputs.append(self._sparse_rows(queries, index_queries, cache, dense))
            output = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, 1)
        elif (causal := self._hip_causal(queries, cache, past + 1)) is not None:
            output = causal
        else:
            visible = past + mx.arange(rows, dtype=mx.int32) + 1
            if total // self.ratio > self.top_k:
                selected = self._select(index_queries, self._pooled(cache), past + 1)
                mask = self._selection_mask(selected, visible, total)
            else:
                mask = token < visible[:, None]
            output = self._masked(queries, cache.keys, cache.values, mask)
        return self.o_proj(output.reshape(1, rows, -1) * mx.sigmoid(gate))

    def _hip_causal(self, queries: mx.array, cache: QSACache, first_visible: int):
        """Causal attention through the HIP kernel when every row fits top_k blocks.

        ``first_visible`` is the number of keys the first query row can see.
        Returns None when a row sees more than ``top_k`` complete blocks.
        """
        kernel = self.kernel
        if not isinstance(kernel, hip.SparseAttention):
            return None
        rows = queries.shape[2]
        if (first_visible + rows - 1) // self.ratio > self.top_k:
            return None
        selected = mx.broadcast_to(
            mx.arange(self.top_k, dtype=mx.int32)[None, :], (rows, self.top_k)
        )
        return kernel.union(
            queries,
            cache.keys,
            cache.values,
            mx.contiguous(selected),
            first_visible,
            kernel.blocked(cache),
        )
