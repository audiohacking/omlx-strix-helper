# SPDX-License-Identifier: Apache-2.0
"""Model families this helper can run.

The same families and contract numbers as the CUDA helper. A family's
``__init__`` needs no GPU library; only workers import its ``stage``.
A host uses this helper for a family only when ``contract`` equals its adapter's.
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


# No family lists an oQh format: the oQh kernels are CUDA sources, so oMLX
# keeps oQh checkpoints on the Mac.
FAMILIES: dict[str, Family] = {
    "qwen4_exp": Family(1, "omlx_cuda_helper.families.qwen4_exp"),
    "minimax_m3_vl": Family(1, "omlx_cuda_helper.families.minimax_m3_vl"),
    "glm5_next": Family(1, "omlx_cuda_helper.families.glm5_next"),
    "deepseek_v41": Family(1, "omlx_cuda_helper.families.deepseek_v41"),
    "qwen3_5": Family(1, "omlx_cuda_helper.families.qwen3_5"),
    "inkling2": Family(1, "omlx_cuda_helper.families.inkling2"),
}


def contracts() -> dict[str, int]:
    return {name: family.contract for name, family in FAMILIES.items()}


def oqh_formats() -> dict[str, int]:
    return {name: f.oqh for name, f in FAMILIES.items() if f.oqh is not None}


def planner(model_type: str) -> ModuleType:
    return importlib.import_module(FAMILIES[model_type].module)


def stage_module(model_type: str) -> ModuleType:
    return importlib.import_module(FAMILIES[model_type].module + ".stage")
