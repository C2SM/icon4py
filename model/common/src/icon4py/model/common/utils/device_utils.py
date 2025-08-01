# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import gt4py._core.definitions as gtx_core_defs  # TODO(havogt): avoid this private import
from gt4py.next import allocators as gtx_allocators, backend as gtx_backend


try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = None


def is_cupy_device(
    allocator: gtx_allocators.FieldBufferAllocationUtil | None,
) -> bool:
    # TODO(havogt): Add to gt4py `gtx_allocators.is_field_buffer_allocation_util_for(...)`
    # and consider exposing CUPY_DEVICE_TYPE or move this function to gt4py.
    if (allocator := gtx_allocators.get_allocator(allocator, default=None)) is not None:
        return allocator.__gt_device_type__ is gtx_core_defs.CUPY_DEVICE_TYPE
    return False


def sync(backend: Optional[gtx_backend.Backend] = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    if is_cupy_device(backend.allocator):
        cp.cuda.runtime.deviceSynchronize()
