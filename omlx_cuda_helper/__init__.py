# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator helper for oMLX on a Strix Halo machine.

The package name and the wire protocol match the CUDA helper, so the oMLX
already running on the Mac drives this process without a Strix-specific path.
"""

__version__ = "0.1.11"

# Framing and ops of the prefill data channel (oMLX ``wire.PROTOCOL``).
DATA_PROTOCOL = 1
# Ops of the control channel between a paired host and this helper.
CONTROL_VERSION = 1
# A stage worker prints this before the JSON that says it is ready.
READY_PREFIX = "OMLX_CUDA_HELPER_READY "
