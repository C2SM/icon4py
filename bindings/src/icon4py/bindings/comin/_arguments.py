# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The inputs of the granule's functions ('_granule.py') as zero-copy views of ICON's variables.

'ArrayParam' describes an array input: its dtype, rank, Field dimensions, memory space,
presence and padding. 'array_argument' builds the value the granule gets from a ComIn variable
of ICON's (a zero-copy view in the layout ICON's Fortran arrays have: the same shape, dtype
and Fortran-order strides, domains starting at 0, a 'gtx.Field' where the input is a Field),
with the rules for absent and empty optional inputs. These are the values py2fgen builds from
the same Fortran arrays, so that the granule gets the same input on both routes (the py2fgen
probe, '_probe.py', compares them).
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any, Final

import numpy as np
from gt4py import next as gtx

from icon4py.bindings.comin import _views


DOMAIN_ID: Final = 1
"""icon4py runs on one domain; the plugin reads ICON's variables of domain 1."""

MAX_RANK: Final = 5
"""ComIn variables are 5-dimensional; ICON pads the unused trailing extents with 1."""

LOCATIONS: Final = ("cell", "edge", "vertex")

_BUFFER_DTYPE: Final[Mapping[np.dtype, np.dtype]] = {
    np.dtype(np.float64): np.dtype(np.float64),
    np.dtype(np.float32): np.dtype(np.float32),
    np.dtype(np.int32): np.dtype(np.int32),
    np.dtype(np.bool_): np.dtype(np.int32),
}
"""The dtype of ICON's buffer of an array input (ComIn has no logical type)."""


@dataclasses.dataclass(frozen=True)
class ArrayParam:
    """An array input of a function of the granule."""

    name: str
    dtype: np.dtype
    """The dtype of the value the granule gets (bool for ICON's logicals)."""
    rank: int
    dims: tuple[gtx.Dimension, ...] | None = None
    """The Field dimensions, or 'None' for an input that is a plain array."""
    host: bool = False
    """A NumPy array also when ICON runs on the GPU (else on the device there)."""
    optional: bool = False
    """ICON may not have the array: the input is then 'None'."""
    location: str | None = None
    """'cell', 'edge' or 'vertex' if the entries along 'axis' beyond the local number of that
    location are padding; 'None' if every entry is real."""
    axis: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.dims is not None and len(self.dims) != self.rank:
            raise ValueError(f"'{self.name}': {len(self.dims)} dimensions for rank {self.rank}.")
        if self.location is not None and self.location not in LOCATIONS:
            raise ValueError(f"'{self.name}': unknown location {self.location!r}.")
        if self.dtype not in _BUFFER_DTYPE:
            raise ValueError(f"'{self.name}': unsupported dtype {self.dtype}.")


Param = ArrayParam | type
"""An input: an array, or a scalar or object of the given type."""


@dataclasses.dataclass(frozen=True)
class Signature:
    """The inputs of one function of the granule, in order."""

    name: str
    function: Callable[..., None]
    params: Mapping[str, Param]

    @functools.cached_property
    def arrays(self) -> tuple[ArrayParam, ...]:
        return tuple(p for p in self.params.values() if isinstance(p, ArrayParam))


@dataclasses.dataclass(frozen=True)
class BoundArray:
    """An array input: ICON's variable (or another buffer) for an array parameter."""

    param: ArrayParam
    variable: Any
    """The 'comin.variable', or a buffer of the plugin's (NumPy or CuPy)."""
    shape: tuple[int, ...]
    """The used extents (the input's shape)."""
    present: bool
    """Presence of an optional input."""


@dataclasses.dataclass(frozen=True)
class BoundFunction:
    signature: Signature
    arrays: tuple[BoundArray, ...]


@dataclasses.dataclass(frozen=True)
class Pointer:
    """An ICON variable as the granule sees it: addresses, used shape, presence."""

    device: int | None
    host: int | None
    shape: tuple[int, ...]
    present: bool


def icon_bound_array(variable: Any, param: ArrayParam, icon_name: str | None = None) -> BoundArray:
    """
    ICON's own variable ('icon_name', default: the input's name) as the array input 'param':
    the first 'rank' extents of ComIn's 5-D shape, so every other extent must be 1, the block
    axis included (one block per patch, nblks == 1). Only valid inside a callback that requested
    it.
    """
    shape = np.asarray(variable).shape  # the host buffer's ('__cuda_array_interface__' too)
    rank = param.rank
    if len(shape) != MAX_RANK or any(e != 1 for e in shape[rank:]):
        raise ValueError(
            f"ICON's variable '{icon_name or param.name}' has the shape {shape}: expected {rank}"
            " used extents and 1 elsewhere (one block, nblks == 1), as the granule's input"
            f" '{param.name}'."
        )
    return BoundArray(param=param, variable=variable, shape=shape[:rank], present=True)


def is_on_device(bound: BoundArray, device_xp: ModuleType | None) -> bool:
    return device_xp is not None and not bound.param.host


def _null_argument(param: ArrayParam) -> None:
    # py2fgen's rule for a NULL address ('py2fgen._conversion.as_array')
    if param.optional:
        return None
    raise ValueError(f"'{param.name}': a NULL address for a non-optional input.")


def _zero_extent_address(bound: BoundArray, device_xp: ModuleType | None) -> int:
    # the adapter builds no buffer on a NULL host address ('comin_var_get_ptr failed')
    try:
        return variable_data_ptr(bound, device_xp)
    except ValueError:
        return 0


def array_argument(bound: BoundArray, device_xp: ModuleType | None) -> Any:
    """
    Build the value of one array input.

    Device inputs are CuPy views of ICON's device array if 'device_xp' is CuPy, host inputs are
    NumPy views; both are sliced from 5 to 'rank' dimensions. A logical input (an INTEGER 0/1
    copy in ICON) becomes a bool copy.

    An optional input whose address is NULL is 'None', any other a view of the given shape, as
    py2fgen passes them ('py2fgen._conversion.as_array'). The address is the device address for
    device inputs on the GPU and the host address otherwise, as ComIn hands out ICON's
    variables. So:
    - an absent optional input (ICON has no such variable) becomes 'None';
    - a zero-extent array becomes 'None' if its address is NULL (on the GPU the device address
      of a zero-size array is NULL, e.g. for an empty 'zd_*' list on a rank), else an empty
      array of its shape;
    - any other array with a NULL address becomes 'None' if optional; else it is an error.
    """
    param = bound.param
    if not bound.present:
        return None
    xp = device_xp if (device_xp is not None and is_on_device(bound, device_xp)) else np
    dtype = _BUFFER_DTYPE[param.dtype]
    rank = param.rank
    if 0 in bound.shape:
        if _zero_extent_address(bound, device_xp) == 0:
            return _null_argument(param)
        array = xp.empty(bound.shape, dtype=dtype, order="F")
    else:
        full = xp.asarray(bound.variable)
        if _views.data_ptr(full) == 0:
            return _null_argument(param)
        array = full[(slice(None),) * rank + (0,) * (MAX_RANK - rank)]
        if array.shape != bound.shape or array.dtype != dtype:
            raise ValueError(
                f"'{param.name}': the view has shape {array.shape} and dtype {array.dtype},"
                f" expected {bound.shape} and {dtype}."
            )
    if param.dtype == np.bool_:
        array = array != 0
    return array if param.dims is None else _views.field_view(array, param.dims)


def variable_data_ptr(bound: BoundArray, device_xp: ModuleType | None) -> int:
    """The address the input is built from: the device pointer on the GPU, else the host one."""
    if is_on_device(bound, device_xp):
        return int(bound.variable.__cuda_array_interface__["data"][0])
    return _views.data_ptr(np.asarray(bound.variable))
