# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (inkling2_loop): the GPU-free part.

A looped checkpoint runs its core layers several times per token. A helper
takes the layer visits up to the end of the first core pass, which reads no
loop messages: there visit ``i`` is layer ``i``, so a stage is the plain
Inkling 2 stage without the loop controllers, and the cut carries what a
plain one does. The host continues the later passes from the cut.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ... import checkpoint
from .. import inkling2 as plain
from ..inkling2 import local_name, text_config

CONTRACT = 1
FIRST_STAGE_MIN = plain.FIRST_STAGE_MIN

# The loop controllers of the core layers; only the later passes read them.
_LOOP = re.compile(r"^layers\.\d+\.loop_")


def owner(tensor: str) -> int | None | bool:
    """Layer of a tensor a stage loads, None for the embedding, else False."""
    if _LOOP.match(local_name(tensor)):
        return False
    return plain.owner(tensor)


def resident(tensor: str) -> bool:
    return owner(tensor) is not False


def loop_end(config: dict[str, Any]) -> int:
    """The first layer past the core: a stage ends at or before it."""
    loop = text_config(config).get("loop_config")
    if not loop:
        raise ValueError("an inkling2_loop checkpoint without a loop_config")
    return int(loop["core_end"])


def plan(model_dir: str | Path) -> tuple[int, list[int]]:
    """(bytes outside the layers, bytes per layer) a helper holds."""
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
    return shared, layers
