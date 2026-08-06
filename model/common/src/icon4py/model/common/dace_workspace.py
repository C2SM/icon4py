# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""External workspace allocation for the DaCe backend.

The DaCe backend in GT4Py can be configured with a
:class:`~gt4py.next.program_processors.runners.dace.transformations.ExternalMemoryAllocator`
to provide workspace memory for transient SDFG arrays
(``transient_memory_mode = EXTERNAL``). The
:class:`IconWorkspaceAllocator` defined here is a process-wide singleton that
caches a single aligned slab per device and reuses it across every compiled
program; :func:`make_custom_dace_backend <icon4py.model.common.model_backends.make_custom_dace_backend>`
installs it by default.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import ModuleType
from typing import ClassVar, Final

import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common.utils import data_allocation


# TODO(edopao): make these configurable via environment variables or model options
_WORKSPACE_SIZE: Final[int] = (
    256 * 1024 * 1024
)  # Max workspace size per device for ICON4Py programs
_WORKSPACE_ALIGNMENT: Final[int] = 256  # Matches DaCe's default transient-storage alignment


def _array_namespace_for(device: gtx.DeviceType) -> ModuleType:
    """Return the array namespace for `device`, requiring cupy on GPU."""
    if device == gtx.CUPY_DEVICE_TYPE:
        try:
            import cupy as cp  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
        except ImportError as err:
            raise RuntimeError(
                f"GPU workspace requested but cupy is not available: {err!r}."
            ) from err
        return cp
    import numpy as np  # noqa: PLC0415 [import-outside-top-level]

    return np


def _aligned_slab(nbytes: int, alignment: int, device: gtx.DeviceType) -> data_allocation.NDArray:
    """Allocate a `nbytes`-byte buffer whose base pointer is `alignment`-aligned.

    The returned slice is the leading `nbytes` bytes of an over-allocated buffer
    offset so that its first element is `alignment`-aligned, satisfying the
    contract enforced by
    `gt4py.next.program_processors.runners.dace.workflow.compilation._validate_external_workspace`.
    """
    xp = _array_namespace_for(device)
    buf = xp.empty(nbytes + alignment, dtype=xp.uint8)
    interface = getattr(buf, "__cuda_array_interface__", None) or getattr(
        buf, "__array_interface__", None
    )
    assert interface is not None, "allocated buffer exposes no array interface"
    ptr = int(interface["data"][0])
    offset = (-ptr) % alignment
    return buf[offset : offset + nbytes]


class IconWorkspaceAllocator:
    """Singleton slab-reuse `ExternalMemoryAllocator` for the DaCe backend.

    Exactly one instance exists per process (enforced by `__new__`); all DaCe
    backends share it via the module-level `ICON_WORKSPACE_ALLOCATOR`. It keeps
    a single private workspace slab per device in `_workspace_slabs`, reused
    across every compiled program.
    """

    _instance: ClassVar[IconWorkspaceAllocator | None] = None
    _workspace_slabs: ClassVar[dict[gtx.DeviceType, data_allocation.NDArray]] = {}

    def __new__(cls) -> IconWorkspaceAllocator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def allocate(
        self, devices: gtx.DeviceType | Iterable[gtx.DeviceType]
    ) -> gtx_transformations.ExternalWorkspace:
        wsp = {}
        if isinstance(devices, gtx.DeviceType):
            devices = [devices]
        for dev in devices:
            if (cached := self._workspace_slabs.get(dev)) is None:
                slab = _aligned_slab(_WORKSPACE_SIZE, _WORKSPACE_ALIGNMENT, dev)
                self._workspace_slabs[dev] = slab
                wsp[dev] = slab
            else:
                wsp[dev] = cached

        return wsp

    def deallocate(self, wsp: gtx_transformations.ExternalWorkspace) -> None:
        # Keep the slab alive for reuse across programs; `deallocate` only
        # signals that this compiled program is done with the workspace.
        pass


ICON_WORKSPACE_ALLOCATOR: Final[IconWorkspaceAllocator] = IconWorkspaceAllocator()
