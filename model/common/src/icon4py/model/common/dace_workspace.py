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

from types import ModuleType
from typing import ClassVar, Final

import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common.utils import data_allocation


_WORKSPACE_SIZE: Final[int] = (
    400 * 1024 * 1024
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

    Implements `gtx_transformations.ExternalMemoryAllocator` structurally:
    `allocate` is called once per SDFG storage type when arguments are
    constructed, and `deallocate` is called once per storage type when the
    compiled program is finalized. The slab is kept alive after `deallocate`
    returns so that subsequent programs reuse it.

    The allocator is part of the DaCe compilation artifact and must be
    picklable. Because `__new__` always returns the existing instance and
    `_workspace_slabs` is a class variable (not part of the instance `__dict__`),
    pickling neither duplicates nor drops the cached slabs: unpickling yields
    the same singleton with its cache intact.
    """

    _instance: ClassVar[IconWorkspaceAllocator | None] = None
    _workspace_slabs: ClassVar[dict[gtx.DeviceType, tuple[data_allocation.NDArray, int]]] = {}

    def __new__(cls) -> IconWorkspaceAllocator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def allocate(
        self, request: gtx_transformations.AllocationRequest
    ) -> gtx_transformations.ExternalWorkspace:
        if request.nbytes > _WORKSPACE_SIZE:
            raise ValueError(
                f"Requested workspace size {request.nbytes} exceeds the maximum "
                f"allowed {_WORKSPACE_SIZE}."
            )
        alignment = max(_WORKSPACE_ALIGNMENT, request.alignment)
        cached = self._workspace_slabs.get(request.device)
        if cached is None or cached[0].nbytes < _WORKSPACE_SIZE or cached[1] < alignment:
            slab = _aligned_slab(_WORKSPACE_SIZE, alignment, request.device)
            self._workspace_slabs[request.device] = (slab, alignment)
            return slab
        return cached[0]

    def deallocate(self, wsp: gtx_transformations.ExternalWorkspace) -> None:
        # Keep the slab alive for reuse across programs; `deallocate` only
        # signals that this compiled program is done with the workspace.
        pass


ICON_WORKSPACE_ALLOCATOR: Final[IconWorkspaceAllocator] = IconWorkspaceAllocator()
