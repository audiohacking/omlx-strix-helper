# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (inkling2_loop): the GPU-free part.

A looped checkpoint runs its core layers several times per token, and every
run (a layer visit) owns its cache entry, so a helper's share counts visits.
A stage holds the layers its visits run, each once, with the core layers'
loop controllers: up to the end of the last core pass, past which the tail
layers start, it needs no layer beyond the core. Every later pass restarts
from the partial sum entering the core and reads the pass before through its
message bank; the cut carries what the host needs to rebuild both, as
``blk.<visit>`` and ``end.<visit>`` (see ``stage``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ... import checkpoint
from .. import inkling2 as plain
from ..inkling2 import text_config

CONTRACT = 1
FIRST_STAGE_MIN = plain.FIRST_STAGE_MIN


def owner(tensor: str) -> int | None | bool:
    """Layer of a tensor a stage loads, None for the embedding, else False."""
    return plain.owner(tensor)


def resident(tensor: str) -> bool:
    return owner(tensor) is not False


def loop(config: dict[str, Any]) -> tuple[int, int, int]:
    """(core start, core end, passes) of a looped checkpoint."""
    found = text_config(config).get("loop_config")
    if not found:
        raise ValueError("an inkling2_loop checkpoint without a loop_config")
    return int(found["core_start"]), int(found["core_end"]), int(found["num_passes"])


def schedule(config: dict[str, Any]) -> list[tuple[int, int]]:
    """``(layer, pass)`` of every layer visit, in the host's order."""
    start, end, passes = loop(config)
    count = text_config(config)["num_hidden_layers"]
    return (
        [(i, 0) for i in range(start)]
        + [(i, p) for p in range(passes) for i in range(start, end)]
        + [(i, 0) for i in range(end, count)]
    )


def loop_end(config: dict[str, Any]) -> int:
    """The first visit of the tail: a stage ends at or before it."""
    start, end, passes = loop(config)
    return start + passes * (end - start)


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer visit) a helper holds: a
    layer's weights count at its first visit, later visits add none."""
    config = checkpoint.load_config(model_dir)
    index = checkpoint.tensors(model_dir)
    count = text_config(config)["num_hidden_layers"]
    shared, layers = 0, [0] * count
    for name, tensor in index.items():
        layer = owner(name)
        if layer is False:
            continue
        size = tensor.nbytes + plain._dense_extra(config, index, name)
        if layer is None:
            shared += size
        elif layer < count:
            layers[layer] += size
    seen: set[int] = set()
    visits = []
    for layer, _ in schedule(config):
        visits.append(0 if layer in seen else layers[layer])
        seen.add(layer)
    return shared, visits
