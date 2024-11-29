# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging as log
from typing import Optional, TypeAlias, Union

import gt4py._core.definitions as gt_core_defs
import gt4py.next as gtx
import numpy as np
from gt4py.next import backend

from icon4py.model.common import dimension, type_alias as ta


""" Enum values from Enum values taken from DLPack reference implementation at:
    https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
    via GT4Py
"""
CUDA_DEVICE_TYPES = (
    gt_core_defs.DeviceType.CUDA,
    gt_core_defs.DeviceType.CUDA_MANAGED,
    gt_core_defs.DeviceType.ROCM,
)


try:
    import cupy as xp
except ImportError:
    import numpy as xp


NDArrayInterface: TypeAlias = Union[np.ndarray, xp.ndarray, gtx.Field]


def as_numpy(array: NDArrayInterface):
    if isinstance(array, np.ndarray):
        return array
    else:
        return array.asnumpy()


def is_cupy_device(backend: backend.Backend) -> bool:
    if backend is not None:
        return backend.allocator.__gt_device_type__ in CUDA_DEVICE_TYPES
    else:
        return False


def array_ns(try_cupy: bool):
    if try_cupy:
        try:
            import cupy as cp

            return cp
        except ImportError:
            log.warn("No cupy installed, falling back to numpy for array_ns")
    import numpy as np

    return np


def import_array_ns(backend: backend.Backend):
    """Import cupy or numpy depending on a chosen GT4Py backend DevicType."""
    return array_ns(is_cupy_device(backend))


def as_field(field: gtx.Field, backend: backend.Backend) -> gtx.Field:
    """Convenience function to transfer an existing Field to a given backend."""
    return gtx.as_field(field.domain, field.ndarray, allocator=backend)


def _size(grid, dim: gtx.Dimension, is_half_dim: bool) -> int:
    if dim == dimension.KDim and is_half_dim:
        return grid.size[dim] + 1
    return grid.size[dim]


def allocate_zero_field(
    *dims: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=ta.wpfloat,
    backend: Optional[backend.Backend] = None,
) -> gtx.Field:
    dimensions = {d: range(_size(grid, d, is_halfdim)) for d in dims}
    return gtx.zeros(dimensions, dtype=dtype, allocator=backend)


def allocate_indices(
    dim: gtx.Dimension,
    grid,
    is_halfdim=False,
    dtype=gtx.int32,
    backend: Optional[backend.Backend] = None,
) -> gtx.Field:
    xp = import_array_ns(backend)
    shapex = _size(grid, dim, is_halfdim)
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype), allocator=backend)
