# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

import gt4py._core.definitions as gt_core_defs
import numpy as np
import numpy.typing as npt
from gt4py import next as gtx
from gt4py.next import backend

from icon4py.model.common import type_alias as ta


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as grid_base


#: Enum values from Enum values taken from DLPack reference implementation at:
#:  https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
#:  via GT4Py
CUDA_DEVICE_TYPES = (
    gt_core_defs.DeviceType.CUDA,
    gt_core_defs.DeviceType.CUDA_MANAGED,
    gt_core_defs.DeviceType.ROCM,
)

try:
    import cupy as xp
except ImportError:
    import numpy as xp

NDArray: TypeAlias = Union[np.ndarray, xp.ndarray]
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


def as_1D_sparse_field(field: gtx.Field | NDArray, target_dim: gtx.Dimension) -> gtx.Field:
    """Convert a 2D sparse field to a 1D flattened (Felix-style) sparse field."""

    buffer = field.ndarray if isinstance(field, gtx.Field) else field
    old_shape = buffer.shape
    assert len(old_shape) == 2
    new_shape = (old_shape[0] * old_shape[1],)
    return gtx.as_field((target_dim,), buffer.reshape(new_shape))


def flatten_first_two_dims(*dims: gtx.Dimension, field: gtx.Field | NDArray) -> gtx.Field:
    """Convert a n-D sparse field to a (n-1)-D flattened (Felix-style) sparse field."""
    buffer = field.ndarray if isinstance(field, gtx.Field) else field
    old_shape = buffer.shape
    assert len(old_shape) >= 2
    flattened_size = old_shape[0] * old_shape[1]
    flattened_shape = (flattened_size,)
    new_shape = flattened_shape + old_shape[2:]
    return gtx.as_field(dims, buffer.reshape(new_shape))


def unflatten_first_two_dims(field: gtx.Field) -> np.array:
    """Convert a (n-1)-D flattened (Felix-style) sparse field back to a n-D sparse field."""
    old_shape = np.asarray(field).shape
    new_shape = (old_shape[0] // 3, 3) + old_shape[1:]
    return np.asarray(field).reshape(new_shape)


def random_field(
    grid,
    *dims,
    low: float = -1.0,
    high: float = 1.0,
    dtype: Optional[npt.DTypeLike] = None,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    backend=None,
) -> gtx.Field:
    arr = np.random.default_rng().uniform(
        low=low, high=high, size=_shape(grid, *dims, extend=extend)
    )
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dims, arr, allocator=backend)


def random_mask(
    grid: grid_base.BaseGrid,
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
    return gtx.as_field(dims, arr)


def zero_field(
    grid: grid_base.BaseGrid,
    *dims: gtx.Dimension,
    dtype=ta.wpfloat,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    backend=None,
) -> gtx.Field:
    field_domain = {dim: (0, stop) for dim, stop in zip(dims, _shape(grid, *dims, extend=extend))}
    return gtx.constructors.zeros(field_domain, dtype=dtype, allocator=backend)


def constant_field(
    grid: grid_base.BaseGrid, value: float, *dims: gtx.Dimension, dtype=ta.wpfloat
) -> gtx.Field:
    return gtx.as_field(
        dims,
        value * np.ones(shape=tuple(map(lambda x: grid.size[x], dims)), dtype=dtype),
    )


def _shape(
    grid,
    *dims: gtx.Dimension,
    extend: Optional[dict[gtx.Dimension, int]] = None,
) -> tuple[int, ...]:
    extend = extend or {}
    return tuple(grid.size[dim] + extend.get(dim, 0) for dim in dims)


def index_field(
    grid,
    dim: gtx.Dimension,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    dtype=gtx.int32,
    backend: Optional[backend.Backend] = None,
) -> gtx.Field:
    xp = import_array_ns(backend)
    shapex = _shape(grid, dim, extend=extend)
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype), allocator=backend)
