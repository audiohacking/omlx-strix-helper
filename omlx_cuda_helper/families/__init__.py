# SPDX-License-Identifier: Apache-2.0
"""Model families this helper can run.

Same contract numbers as the CUDA helper. Only families whose stages exist
are listed, so oMLX keeps the others on the Mac.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType


@dataclass(frozen=True)
class Family:
    contract: int
    module: str
    oqh: int | None = None


FAMILIES: dict[str, Family] = {
    "qwen3_5": Family(1, "omlx_cuda_helper.families.qwen3_5"),
}


def contracts() -> dict[str, int]:
    return {name: family.contract for name, family in FAMILIES.items()}


def oqh_formats() -> dict[str, int]:
    return {name: f.oqh for name, f in FAMILIES.items() if f.oqh is not None}


def planner(model_type: str) -> ModuleType:
    return importlib.import_module(FAMILIES[model_type].module)


def stage_module(model_type: str) -> ModuleType:
    return importlib.import_module(FAMILIES[model_type].module + ".stage")
