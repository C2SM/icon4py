# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import logging as log
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeAlias, TypeGuard, TypeVar

import gt4py.next.typing as gtx_typing
import numpy as np
import numpy.typing as npt
from gt4py import next as gtx
from gt4py.next import allocators as gtx_allocators

from icon4py.model.common import dimension as dims  # noqa: F401 # used in eval by make_fields
from icon4py.model.common.type_alias import vpfloat, wpfloat  # noqa: F401
from icon4py.model.common.utils import device_utils


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as grid_base


try:
    import cupy as xp  # type: ignore[import-not-found]
except ImportError:
    import numpy as xp

ScalarT = TypeVar("ScalarT", bound=gtx_typing.Scalar)
NDArray: TypeAlias = (
    np.ndarray[tuple[int, ...], np.dtype[ScalarT]] | xp.ndarray[tuple[int, ...], np.dtype[ScalarT]]
)
NDArrayInterface: TypeAlias = np.ndarray | xp.ndarray | gtx.Field

ScalarLikeArray: TypeAlias = (
    np.ndarray[tuple[()], np.dtype[ScalarT]] | xp.ndarray[tuple[()], np.dtype[ScalarT]]
)


def is_ndarray(obj: Any) -> TypeGuard[NDArray]:
    return isinstance(obj, (np.ndarray, xp.ndarray))


def is_rank0_ndarray(obj: Any) -> TypeGuard[ScalarLikeArray]:
    return is_ndarray(obj) and obj.shape == ()


def backend_name(backend: gtx_typing.Backend | None) -> str:
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


def scalar_like_array(
    value: ScalarT,
    allocator: ModuleType | gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> ScalarLikeArray[ScalarT]:  # type: ignore[type-var] # ScalarT is a subtype of already specified other types
    """Create a 0-d array (scalar-like) with given value on specified array namespace or allocator."""
    array_ns = allocator if allocator in (np, xp) else import_array_ns(allocator)
    assert array_ns is not None and hasattr(array_ns, "asarray")
    return array_ns.asarray(value)


def as_field(
    field: gtx.Field,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
    embedded_on_host: bool = False,
) -> gtx.Field:
    """Convenience function to transfer an existing Field to a given backend."""
    data = field.asnumpy() if embedded_on_host else field.ndarray
    return gtx.as_field(field.domain, data=data, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_field(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    low: float = -1.0,
    high: float = 1.0,
    dtype: npt.DTypeLike = wpfloat,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    arr = np.random.default_rng().uniform(
        low=low, high=high, size=_shape(grid, *dimensions, extend=extend)
    )
    arr = arr.astype(dtype)
    return gtx.as_field(dimensions, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_sign(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    dtype: npt.DTypeLike | None = None,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    """Generate a random field with values -1 or 1."""
    arr = np.random.default_rng().choice([-1, 1], size=_shape(grid, *dimensions, extend=extend))
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dimensions, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_mask(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    dtype: npt.DTypeLike | None = None,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_typing.Backend | None = None,
) -> gtx.Field:
    rng = np.random.default_rng()
    shape = _shape(grid, *dimensions, extend=extend)
    arr = np.full(shape, False).flatten()
    num_true = int(arr.size * 0.5)
    arr[:num_true] = True
    rng.shuffle(arr)
    arr = np.reshape(arr, newshape=shape)
    if dtype:
        arr = arr.astype(dtype)
    return gtx.as_field(dimensions, arr, allocator=allocator)  # type: ignore [arg-type] # type "ndarray[Any, Any] | NDArrayObject"; expected "NDArrayObject"


def random_ikoffset(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    dtype=gtx.int32,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
):
    ikoffset = empty_field(grid, *dimensions, dtype=dtype, extend=extend, allocator=allocator)
    rng = np.random.default_rng()
    for k in range(grid.num_levels):
        # construct offsets that reach all k-levels except the last (because we are using the entries of this field with `+1`)
        ikoffset.ndarray[:, :, k] = rng.integers(  # type: ignore[index]
            low=0 - k,
            high=grid.num_levels - k - 1,
            size=(ikoffset.shape[0], ikoffset.shape[1]),
        )
    return ikoffset


def empty_field(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    dtype=wpfloat,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    field_domain = {
        dim: (0, stop) for dim, stop in zip(dimensions, _shape(grid, *dimensions, extend=extend))
    }
    return gtx.constructors.empty(field_domain, dtype=dtype, allocator=allocator)


def zero_field(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    dtype=wpfloat,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    field_domain = {
        dim: (0, stop) for dim, stop in zip(dimensions, _shape(grid, *dimensions, extend=extend))
    }
    return gtx.constructors.zeros(field_domain, dtype=dtype, allocator=allocator)


def constant_field(
    grid: grid_base.Grid,
    value: float,
    *dimensions: gtx.Dimension,
    dtype=wpfloat,
    extend: dict[gtx.Dimension, int] | None = None,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    field_domain = {
        dim: (0, stop) for dim, stop in zip(dimensions, _shape(grid, *dimensions, extend=extend))
    }
    return gtx.constructors.full(field_domain, value, dtype=dtype, allocator=allocator)


def _shape(
    grid: grid_base.Grid,
    *dimensions: gtx.Dimension,
    extend: dict[gtx.Dimension, int] | None = None,
) -> tuple[int, ...]:
    extend = extend or {}
    return tuple(grid.size[dim] + extend.get(dim, 0) for dim in dimensions)


def index_field(
    grid: grid_base.Grid,
    dim: gtx.Dimension,
    extend: dict[gtx.Dimension, int] | None = None,
    dtype: npt.DTypeLike = gtx.int32,  # type: ignore [attr-defined]
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> gtx.Field:
    xp = import_array_ns(allocator)
    shapex = _shape(grid, dim, extend=extend)[0]
    return gtx.as_field((dim,), xp.arange(shapex, dtype=dtype), allocator=allocator)


# load variable properties (dtype, field dimensions and extend) from json
p = Path(__file__).resolve().parent / "variable_properties.json"
with p.open("r", encoding="utf-8") as fh:
    _variable_properties = json.load(fh)


def make_fields(varnames, gen_fct, *args, **kwargs) -> dict[str, gtx.Field]:
    fields = {}
    for var in varnames:
        dtype, dimensions, extend = eval(_variable_properties[var])
        this_field = gen_fct(*args, *dimensions, dtype=dtype, extend=extend, **kwargs)
        fields[var] = this_field
    return fields


def get_random_fields(
    grid: grid_base.Grid,
    varnames: list,
    low: float = -1.0,
    high: float = 1.0,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> dict[str, gtx.Field]:
    return make_fields(varnames, random_field, grid, low=low, high=high, allocator=allocator)


def get_zero_fields(
    grid, varnames: list, allocator: gtx_allocators.FieldBufferAllocationUtil | None = None
) -> dict[str, gtx.Field]:
    return make_fields(varnames, zero_field, grid, allocator=allocator)


def get_const_fields(
    grid: grid_base.Grid,
    varnames: list,
    value,
    allocator: gtx_allocators.FieldBufferAllocationUtil | None = None,
) -> dict[str, gtx.Field]:
    return make_fields(varnames, constant_field, grid, value, allocator=allocator)
