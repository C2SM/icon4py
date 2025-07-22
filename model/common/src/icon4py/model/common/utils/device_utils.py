# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py.next import backend as gtx_backend

from icon4py.model.common.utils import (
    data_allocation as data_alloc,
)


try:
    import cupy as cp
except ImportError:
    cp = None


def sync(backend: Optional[gtx_backend.Backend] = None) -> None:
    """
    Synchronize the device if appropriate for the given backend.

    Note: this is and ad-hoc interface, maybe the function should get the device to sync for.
    """
    if data_alloc.is_cupy_device(backend):
        cp.cuda.runtime.deviceSynchronize()
