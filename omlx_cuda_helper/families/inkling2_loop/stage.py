# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Inkling 2 Mini: layer visits ``[first, last)``.

Visit ``i`` runs layer ``schedule[i]``; the stage holds each of those layers
once, with the core layers' loop controllers. As the host's model does, the
first core visit keeps the prelude (the partial sum entering the core and
the blocks before it); each later pass restarts from it, and the blocks the
pass before completed, with the partial sum it ended on, are that pass's
message bank.

The cut carries the partial sum ``h``, ``blk.<visit>`` for every boundary
visit before it (a later pass's first visit takes the prelude's sum, which
``blk.<core start>`` already holds) and ``end.<visit>``, the partial sum a
pass ended on, for every later pass that starts before the cut. A stage past
the first core pass starts at visit 0.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import from_host, to_host
from ..inkling2.layers import unit_rms
from ..inkling2.stage import Arrays
from ..inkling2.stage import Stage as LayerStage
from . import loop, loop_end, owner, schedule


class Stage(LayerStage):
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        self.schedule = schedule(config)
        self.core, core_end, _ = loop(config)
        end = loop_end(config)
        if last > end:
            raise ValueError(
                f"An Inkling 2 Mini stage ends by visit {end}, the end of the "
                f"last core pass; it was asked for visits up to {last}"
            )
        if first and last > core_end:
            raise ValueError(
                "An Inkling 2 Mini stage past the first core pass starts at visit 0"
            )
        held = [layer for layer, _ in self.schedule[first:last]]
        self.base = min(held)
        super().__init__(model_dir, self.base, max(held) + 1, owner=owner)
        self.first, self.last = first, last
        if last > core_end and any(
            self.layer(v).loop is None for v in range(core_end, last)
        ):
            raise ValueError(
                "This Inkling 2 Mini checkpoint keeps its loop controllers in the "
                "published layout; a helper takes visits of the first pass only"
            )

    def layer(self, visit: int):
        return self.layers[self.schedule[visit][0] - self.base]

    def _uses(self, layer: int, visits: range) -> bool:
        return any(self.schedule[v][0] == layer for v in visits)

    def drop_layer(self) -> None:
        """Give up the last visit, and its layer when no earlier visit runs it."""
        self.last -= 1
        layer = self.schedule[self.last][0]
        top = self.base + len(self.layers) - 1
        if layer == top and not self._uses(layer, range(self.first, self.last)):
            self.layers.pop()
            mx.clear_cache()

    def add_layer(self) -> None:
        """Take the visit after the last one, loading its layer if not held."""
        layer = self.schedule[self.last][0]
        if layer == self.base + len(self.layers):
            self.layers.append(self._load_layer(layer))
        self.last += 1

    def load_prefix(self, job: list, state: Arrays) -> None:
        for j, cache in enumerate(job):
            visit = self.first + j
            name = f"{visit}."
            if name + "k" in state:
                cache.append(from_host(state[name + "k"]), from_host(state[name + "v"]))
                if self.layer(visit).window:
                    cache.keep(self.layer(visit).window)
            for slot in range(4):
                if f"{name}c{slot}" in state:
                    cache.conv[slot] = from_host(state[f"{name}c{slot}"])
        mx.eval([a for cache in job for a in cache.arrays()])

    def _forward(self, job, h, entered, keep, ends=None):
        """Run the job's visits on ``h``. ``entered`` maps each boundary visit
        to the partial sum it took and ``ends`` (when given) each later pass's
        first visit to the partial sum the pass before ended on. Returns the
        partial sum, the chunk's state and, for each visit in ``keep``, the
        partial sum entering it from that chunk row on."""
        blocks = [unit_rms(entered[i]) for i in sorted(entered)]
        state: dict[str, mx.array] = {}
        kept: dict[int, mx.array] = {}
        prelude = bank = None
        for j, cache in enumerate(job):
            visit = self.first + j
            index, loop_pass = self.schedule[visit]
            layer = self.layer(visit)
            if index == self.core and prelude is None:
                prelude = (h, list(blocks))
            elif index == self.core:
                if ends is not None:
                    ends[visit] = h
                bank = (blocks[len(prelude[1]) + 1 :], unit_rms(h))
                h, blocks = prelude[0], list(prelude[1])
            if keep and visit in keep:
                kept[visit] = h[:, keep[visit] :]
            if layer.boundary and not (index == self.core and loop_pass):
                entered[visit] = h
            h = layer(h, blocks, cache, bank if loop_pass else None)
            state[f"{visit}.k"], state[f"{visit}.v"] = cache.appended
            for slot, value in enumerate(cache.conv):
                state[f"{visit}.c{slot}"] = value
            # Bound the lazy graph (and its temporaries) to one visit.
            mx.eval(h, blocks, *state.values(), *cache.arrays(), *kept.values())
        return h, state, kept

    def warm(self, tokens: int) -> float:
        self.enable_kernels()
        ids = mx.arange(tokens, dtype=mx.int32) % (self.cfg.vocab - 3) + 3
        if self.first:
            h = mx.random.normal((1, tokens, self.cfg.hidden)).astype(self.dtype)
            entered = {i: h for i in range(0, self.first, self.cfg.block)}
        else:
            h, entered = self._embed(ids)[None], {}
        mx.eval(h)
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(self.last - self.first), h, dict(entered), None)
            mx.synchronize()
            seconds = time.perf_counter() - began
        mx.clear_cache()
        return seconds

    def run(
        self,
        job: list,
        *,
        start: int,
        tokens: np.ndarray,
        boundary: Arrays | None,
        inputs: Arrays,
        following: int,
        capture: dict | None,
    ) -> tuple[Arrays, Arrays]:
        received: Arrays = {}
        if boundary is None:
            h = self._embed(mx.array(np.asarray(tokens, dtype=np.int32)))[None]
        else:
            h = from_host(boundary["h"])
            received = {n: v for n, v in boundary.items() if n.startswith("blk.")}
        entered = {int(n[4:]): from_host(v) for n, v in received.items()}
        keep = None
        if capture and start + h.shape[1] > capture["from"]:
            row = max(0, capture["from"] - start)
            keep = {i: row for i in capture["layers"]}
        ends: dict[int, mx.array] = {}
        h, state, kept = self._forward(job, h, entered, keep, ends)
        out = {f"cap.{i}": to_host(rows) for i, rows in kept.items()}
        out["h"] = to_host(h)
        # The boundaries before this stage travel on as they came.
        out.update(received)
        for i, value in entered.items():
            if f"blk.{i}" not in received:
                out[f"blk.{i}"] = to_host(value)
        for i, value in ends.items():
            out[f"end.{i}"] = to_host(value)
        return out, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
