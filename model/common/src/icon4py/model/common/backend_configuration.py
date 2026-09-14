# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""External workspace allocation for the DaCe backend.

The DaCe backend in GT4Py can be configured with an
:class:`~gt4py.next.program_processors.runners.dace.workflow.common.ExternalWorkspace`
to provide workspace memory for transient SDFG arrays
(``transient_memory_mode = EXTERNAL``). When a :class:`BackendConfig` is
provided, :func:`get_dace_options <icon4py.model.common.model_options.get_dace_options>`
calls the :class:`IconWorkspaceAllocator` defined here — a process-wide
singleton that caches a single workspace slab per device and reuses it across
every compiled program.

The size of the workspace is configurable per experiment via :class:`BackendConfig`
(see :func:`backend_config_from_env` for an environment-variable based default).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable
from typing import ClassVar, Final

import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon

from icon4py.model.common.utils import data_allocation


@dataclasses.dataclass(frozen=True, kw_only=True)
class BackendConfig:
    """External DaCe workspace sizing, configurable per experiment."""

    #: Workspace size in bytes, per device.
    workspace_size: int

    def __post_init__(self) -> None:
        if self.workspace_size <= 0:
            raise ValueError(f"'workspace_size' must be positive, got {self.workspace_size}.")


def backend_config_from_env() -> BackendConfig | None:
    """Build a :class:`BackendConfig` from environment variables.

    Reads ``ICON4PY_BACKEND_WORKSPACE_SIZE`` and returns ``None`` when it is
    not set.
    """
    size = os.environ.get("ICON4PY_BACKEND_WORKSPACE_SIZE")
    if size is None:
        return None
    return BackendConfig(workspace_size=int(size))


def _get_slab(nbytes: int, device: gtx.DeviceType) -> data_allocation.NDArray:
    """Allocate a `nbytes`-byte buffer allocated on ``device``."""
    xp = data_allocation.array_ns(use_cupy=(device != gtx.DeviceType.CPU))
    return xp.empty(nbytes, dtype=xp.uint8)


class IconWorkspaceAllocator:
    """Singleton workspace allocator for the DaCe backend.

    Exactly one instance exists per process (enforced by `__new__`); all DaCe
    backends share it via the module-level `ICON_WORKSPACE_ALLOCATOR`. It keeps
    a single private workspace slab per device in `_workspace_slabs`, reused
    across every compiled program. On a cache hit the slab's size is validated
    against the value passed to `allocate`.
    """

    _instance: ClassVar[IconWorkspaceAllocator | None] = None
    _workspace_slabs: ClassVar[dict[gtx.DeviceType, data_allocation.NDArray]] = {}

    def __new__(cls) -> IconWorkspaceAllocator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def allocate(
        self,
        devices: gtx.DeviceType | Iterable[gtx.DeviceType],
        *,
        size: int,
    ) -> gtx_wfdcommon.ExternalWorkspace:
        if isinstance(devices, gtx.DeviceType):
            devices = [devices]
        wsp: gtx_wfdcommon.ExternalWorkspace = {}
        for dev in devices:
            if (cached := self._workspace_slabs.get(dev)) is not None:
                if cached.nbytes != size:
                    raise ValueError(
                        f"Workspace size mismatch for {dev!s}: cached slab has "
                        f"{cached.nbytes} bytes but 'allocate' was called with "
                        f"size={size}."
                    )
                wsp[dev] = cached
            else:
                slab = _get_slab(size, dev)
                self._workspace_slabs[dev] = slab
                wsp[dev] = slab
        return wsp


ICON_WORKSPACE_ALLOCATOR: Final[IconWorkspaceAllocator] = IconWorkspaceAllocator()
