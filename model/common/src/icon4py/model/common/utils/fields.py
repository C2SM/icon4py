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
from gt4py.next import backend
import numpy as np
import numpy.typing as npt

from icon4py.model.common import dimension, type_alias as ta
from icon4py.model.common.grid.base import BaseGrid

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
            log.warning("No cupy installed, falling back to numpy for array_ns")
    import numpy as np

    return np


def import_array_ns(backend: backend.Backend):
    """Import cupy or numpy depending on a chosen GT4Py backend DevicType."""
    return array_ns(is_cupy_device(backend))


def as_field(field: gtx.Field, backend: backend.Backend) -> gtx.Field:
    """Convenience function to transfer an existing Field to a given backend."""
    return gtx.as_field(field.domain, field.ndarray, allocator=backend)


def as_1D_sparse_field(field: gtx.Field, target_dim: gtx.Dimension) -> gtx.Field:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    buffer = field.ndarray
    return numpy_to_1D_sparse_field(buffer, target_dim)


def numpy_to_1D_sparse_field(field: np.ndarray, dim: gtx.Dimension) -> gtx.Field:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""
    old_shape = field.shape
    assert len(old_shape) == 2
    new_shape = (old_shape[0] * old_shape[1],)
    return as_field((dim,), field.reshape(new_shape))


def flatten_first_two_dims(*dims: gtx.Dimension, field: gtx.Field) -> gtx.Field:
    """Convert a n-D sparse field to a (n-1)-D flattened (Felix-style) sparse field."""
    buffer = field.ndarray
    old_shape = buffer.shape
    assert len(old_shape) >= 2
    flattened_size = old_shape[0] * old_shape[1]
    flattened_shape = (flattened_size,)
    new_shape = flattened_shape + old_shape[2:]
    newarray = buffer.reshape(new_shape)
    return as_field(dims, newarray)


def unflatten_first_two_dims(field: gtx.Field) -> np.array:
    """Convert a (n-1)-D flattened (Felix-style) sparse field back to a n-D sparse field."""
    old_shape = np.asarray(field).shape
    new_shape = (old_shape[0] // 3, 3) + old_shape[1:]
    return np.asarray(field).reshape(new_shape)


def _size(grid, dim: gtx.Dimension, is_half_dim: bool) -> int:
    if dim == dimension.KDim and is_half_dim:
        return grid.size[dim] + 1
    return grid.size[dim]



def random_field(
    grid,
    *dims,
    low: float = -1.0,
    high: float = 1.0,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    dtype: Optional[npt.DTypeLike] = None,
) -> gtx.Field:
    arr = np.random.default_rng().uniform(
        low=low, high=high, size=_shape(grid, *dims, extend=extend)
    )
    if dtype:
        arr = arr.astype(dtype)
    return as_field(dims, arr)

def zero_field(
    grid: BaseGrid,
    *dims: gtx.Dimension,
    dtype=ta.wpfloat,
    extend: Optional[dict[gtx.Dimension, int]] = None,
) -> gtx.Field:
    return as_field(dims, xp.zeros(shape=_shape(grid, *dims, extend=extend), dtype=dtype))


def constant_field(
    grid: BaseGrid, value: float, *dims: gtx.Dimension, dtype=ta.wpfloat
) -> gtx.Field:
    return as_field(
        dims,
        value * np.ones(shape=tuple(map(lambda x: grid.size[x], dims)), dtype=dtype),
    )



def _shape(
    grid,
    *dims: gtx.Dimension,
    extend: Optional[dict[gtx.Dimension, int]] = None,
):
    extend = extend or {}
    return tuple(grid.size[dim] + extend.get(dim, 0) for dim in dims)


def random_mask(
    grid: BaseGrid,
    *dims: gtx.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
    extend: Optional[dict[gtx.Dimension, int]] = None,
) -> gtx.Field:
    rng = np.random.default_rng()
    shape = _shape(grid, *dims, extend=extend)
    arr = np.full(shape, False).flatten()
    num_true = int(arr.size * 0.5)
    arr[:num_true] = True
    rng.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if dtype:
        arr = arr.astype(dtype)
    return as_field(dims, arr)



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


