# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging as log
from types import ModuleType
from typing import TYPE_CHECKING, Optional, TypeAlias, Union

import numpy as np
import numpy.typing as npt
from gt4py import next as gtx
from gt4py.next import allocators as gtx_allocators, backend as gtx_backend

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import device_utils


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as grid_base


try:
    import cupy as xp
except ImportError:
    import numpy as xp

NDArray: TypeAlias = Union[np.ndarray, xp.ndarray]
NDArrayInterface: TypeAlias = Union[np.ndarray, xp.ndarray, gtx.Field]


def backend_name(backend: gtx_backend.Backend | None) -> str:
    return "embedded" if backend is None else backend.name


def as_numpy(array: NDArrayInterface) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, gtx.Field):
        return array.asnumpy()
    else:
        import cupy as cp

        return cp.asnumpy(array)


def array_ns(try_cupy: bool) -> ModuleType:
    if try_cupy:
        try:
            import cupy as cp

            return cp
        except ImportError:
            log.warning("No cupy installed, falling back to numpy for array_ns")
    import numpy as np

    return np


def import_array_ns(allocator: gtx_allocators.FieldBufferAllocationUtil | None) -> ModuleType:
    """Import cupy or numpy depending on a chosen GT4Py backend DevicType."""
    return array_ns(device_utils.is_cupy_device(allocator))


def as_field(
    field: gtx.Field,
    backend: Optional[gtx_backend.Backend] = None,
    embedded_on_host: bool = False,
) -> gtx.Field:
    """Convenience function to transfer an existing Field to a given backend."""
    data = field.asnumpy() if embedded_on_host else field.ndarray
    return gtx.as_field(field.domain, data=data, allocator=backend)


def flatten_first_two_dims(
    *dims: gtx.Dimension, field: gtx.Field | NDArray, backend: Optional[gtx_backend.Backend] = None
) -> gtx.Field:
    """Convert a n-D sparse field or ndarray to a (n-1)-D flattened (Felix-style) sparse field."""
    buffer = field.ndarray if isinstance(field, gtx.Field) else field
    old_shape = buffer.shape
    assert len(old_shape) >= 2
    flattened_size = old_shape[0] * old_shape[1]
    flattened_shape = (flattened_size,)
    new_shape = flattened_shape + old_shape[2:]
    return gtx.as_field(dims, buffer.reshape(new_shape), allocator=backend)


def unflatten_first_two_dims(field: gtx.Field | NDArray) -> NDArray:
    """Convert a (n-1)-D flattened (Felix-style) sparse field or ndarray to a n-D sparse NDArray."""
    buffer = field.ndarray if isinstance(field, gtx.Field) else field
    old_shape = buffer.shape
    new_shape = (old_shape[0] // 3, 3) + old_shape[1:]
    return buffer.reshape(new_shape)


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
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    dtype: Optional[npt.DTypeLike] = None,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    backend: Optional[gtx_backend.Backend] = None,
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
    return gtx.as_field(dims, arr, allocator=backend)


def zero_field(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    dtype=ta.wpfloat,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    backend=None,
) -> gtx.Field:
    field_domain = {dim: (0, stop) for dim, stop in zip(dims, _shape(grid, *dims, extend=extend))}
    return gtx.constructors.zeros(field_domain, dtype=dtype, allocator=backend)


def constant_field(
    grid: grid_base.Grid,
    value: float,
    *dims: gtx.Dimension,
    dtype=ta.wpfloat,
    backend=None,
) -> gtx.Field:
    return gtx.as_field(
        dims,
        value * np.ones(shape=tuple(map(lambda x: grid.size[x], dims)), dtype=dtype),
        allocator=backend,
    )


def _shape(
    grid: grid_base.Grid,
    *dims: gtx.Dimension,
    extend: Optional[dict[gtx.Dimension, int]] = None,
) -> tuple[int, ...]:
    extend = extend or {}
    return tuple(grid.size[dim] + extend.get(dim, 0) for dim in dims)


def index_field(
    grid: grid_base.Grid,
    dim: gtx.Dimension,
    extend: Optional[dict[gtx.Dimension, int]] = None,
    dtype=gtx.int32,
    backend: Optional[gtx_backend.Backend] = None,
) -> gtx.Field:
    xp = import_array_ns(backend)
    shapex = _shape(grid, dim, extend=extend)[0]
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype), allocator=backend)
