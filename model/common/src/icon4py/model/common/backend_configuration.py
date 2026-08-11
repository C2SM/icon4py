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

The size and alignment of the workspace are configurable per experiment via
:class:`BackendConfig` (see :func:`backend_config_from_env` for an
environment-variable based default).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable
from types import ModuleType
from typing import ClassVar, Final

import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from icon4py.model.common.utils import data_allocation


_DEFAULT_ALIGNMENT: Final[int] = 256  # Matches DaCe's default transient-storage alignment


@dataclasses.dataclass(frozen=True, kw_only=True)
class BackendConfig:
    """External DaCe workspace sizing, configurable per experiment."""

    #: Workspace size in bytes, per device.
    workspace_size: int
    #: Base-pointer alignment in bytes. Must be a positive power of two.
    workspace_alignment: int

    def __post_init__(self) -> None:
        if self.workspace_size <= 0:
            raise ValueError(f"'workspace_size' must be positive, got {self.workspace_size}.")
        if (
            self.workspace_alignment <= 0
            or (self.workspace_alignment & (self.workspace_alignment - 1)) != 0
        ):
            raise ValueError(
                f"'workspace_alignment' must be a positive power of two, got {self.workspace_alignment}."
            )


def backend_config_from_env() -> BackendConfig | None:
    """Build a :class:`BackendConfig` from environment variables.

    Reads ``ICON4PY_BACKEND_WORKSPACE_SIZE`` and (optionally)
    ``ICON4PY_BACKEND_WORKSPACE_ALIGNMENT``. Returns ``None`` when
    ``ICON4PY_BACKEND_WORKSPACE_SIZE`` is not set. When
    ``ICON4PY_BACKEND_WORKSPACE_ALIGNMENT`` is not set, :data:`_DEFAULT_ALIGNMENT`
    is used.
    """
    size = os.environ.get("ICON4PY_BACKEND_WORKSPACE_SIZE")
    if size is None:
        return None
    alignment = os.environ.get("ICON4PY_BACKEND_WORKSPACE_ALIGNMENT")
    return BackendConfig(
        workspace_size=int(size),
        workspace_alignment=int(alignment) if alignment is not None else _DEFAULT_ALIGNMENT,
    )


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


def _array_base_ptr(buf: data_allocation.NDArray) -> int:
    """Return the base pointer of `buf` from its array interface."""
    interface = getattr(buf, "__cuda_array_interface__", None) or getattr(
        buf, "__array_interface__", None
    )
    assert interface is not None, "allocated buffer exposes no array interface"
    return int(interface["data"][0])


def _aligned_slab(nbytes: int, alignment: int, device: gtx.DeviceType) -> data_allocation.NDArray:
    """Allocate a `nbytes`-byte buffer whose base pointer is `alignment`-aligned.

    The returned slice is the leading `nbytes` bytes of an over-allocated buffer
    offset so that its first element is `alignment`-aligned, satisfying the
    contract enforced by
    `gt4py.next.program_processors.runners.dace.workflow.compilation._validate_external_workspace`.
    """
    xp = _array_namespace_for(device)
    buf = xp.empty(nbytes + alignment, dtype=xp.uint8)
    ptr = _array_base_ptr(buf)
    offset = (-ptr) % alignment
    return buf[offset : offset + nbytes]


class IconWorkspaceAllocator:
    """Singleton slab-reuse `ExternalMemoryAllocator` for the DaCe backend.

    Exactly one instance exists per process (enforced by `__new__`); all DaCe
    backends share it via the module-level `ICON_WORKSPACE_ALLOCATOR`. It keeps
    a single private workspace slab per device in `_workspace_slabs`, reused
    across every compiled program. On a cache hit the slab's size and base-
    pointer alignment are validated against the values passed to `allocate`.
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
        alignment: int = _DEFAULT_ALIGNMENT,
    ) -> gtx_transformations.ExternalWorkspace:
        if isinstance(devices, gtx.DeviceType):
            devices = [devices]
        wsp = {}
        for dev in devices:
            if (cached := self._workspace_slabs.get(dev)) is not None:
                if cached.nbytes != size:
                    raise ValueError(
                        f"Workspace size mismatch for {dev!s}: cached slab has "
                        f"{cached.nbytes} bytes but 'allocate' was called with "
                        f"size={size}."
                    )
                if (_array_base_ptr(cached) % alignment) != 0:
                    raise ValueError(
                        f"Workspace alignment mismatch for {dev!s}: cached slab base "
                        f"pointer is not {alignment}-aligned."
                    )
                wsp[dev] = cached
            else:
                slab = _aligned_slab(size, alignment, dev)
                self._workspace_slabs[dev] = slab
                wsp[dev] = slab
        return wsp

    def deallocate(self, wsp: gtx_transformations.ExternalWorkspace) -> None:
        # Keep the slab alive for reuse across programs; `deallocate` only
        # signals that this compiled program is done with the workspace.
        pass


ICON_WORKSPACE_ALLOCATOR: Final[IconWorkspaceAllocator] = IconWorkspaceAllocator()
