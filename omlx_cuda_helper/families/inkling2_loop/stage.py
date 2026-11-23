# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Inkling 2 Mini: layers ``[first, last)`` of the first pass.

Before the end of the first core pass every layer runs once, so this is the
Inkling 2 stage loading no loop controllers.
"""

from __future__ import annotations

from pathlib import Path

from ... import checkpoint
from ..inkling2.stage import Stage
from . import loop_end, owner


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    end = loop_end(checkpoint.load_config(model_dir))
    if last > end:
        raise ValueError(
            f"An Inkling 2 Mini stage ends by layer {end}, the end of the first "
            f"core pass; it was asked for layers up to {last}"
        )
    return Stage(model_dir, first, last, owner=owner)
