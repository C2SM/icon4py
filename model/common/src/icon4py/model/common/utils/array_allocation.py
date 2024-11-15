# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
from gt4py._core import definitions as core_defs
from gt4py.next import allocators, backend as gt4py_backend

from icon4py.model.common import field_type_aliases as fa


log = logging.getLogger(__name__)


def is_cupy_device(backend: gt4py_backend.Backend) -> bool:
    return (
        allocators.is_field_allocator_for(backend.allocator, core_defs.DeviceType.CUDA)
        or allocators.is_field_allocator_for(backend.allocator, core_defs.DeviceType.CUDA_MANAGED)
        or allocators.is_field_allocator_for(backend.allocator, core_defs.DeviceType.ROCM)
    )


def array_ns(try_cupy: bool):
    if try_cupy:
        try:
            import cupy as cp

            return cp
        except ImportError:
            log.warning("No cupy installed falling back to numpy for array_ns")
    return np


def array_ns_from_obj(array: fa.AnyNDArray):
    if isinstance(array, np.ndarray):
        return np
    else:
        import cupy as cp

        return cp
