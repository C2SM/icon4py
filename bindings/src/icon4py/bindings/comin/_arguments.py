# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
The arguments of the py2fgen-exported functions, as py2fgen builds them, from ICON's variables.

'signature' derives the parameters of an exported function from its 'param_descriptors' and
its type hints. 'array_argument' builds the Python argument py2fgen would pass for an array
parameter from a ComIn variable of ICON's (a zero-copy view in py2fgen's layout: the same
shape, dtype and Fortran-order strides, domains starting at 0, a 'gtx.Field' only for a
Field-annotated parameter), with py2fgen's rules for absent and empty optional arguments.
"""

import dataclasses
import typing
from collections.abc import Callable
from types import ModuleType
from typing import Any, Final

import numpy as np
from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings import icon4py_export
from icon4py.bindings.comin import _views
from icon4py.tools import py2fgen


DOMAIN_ID: Final = 1
"""icon4py runs on one domain; the plugin reads ICON's variables of domain 1."""

MAX_RANK: Final = 5
"""ComIn variables are 5-dimensional; ICON pads the unused trailing extents with 1."""

_BUFFER_DTYPE: Final[dict[ts.ScalarKind, np.dtype]] = {
    py2fgen.FLOAT64: np.dtype(np.float64),
    py2fgen.FLOAT32: np.dtype(np.float32),
    py2fgen.INT32: np.dtype(np.int32),
    py2fgen.BOOL: np.dtype(np.int32),
}
"""The dtype of ICON's buffer of an array parameter (ComIn has no logical type)."""


@dataclasses.dataclass(frozen=True)
class ArrayParam:
    name: str
    descriptor: py2fgen.ArrayParamDescriptor
    dims: tuple[gtx.Dimension, ...] | None
    """The Field dimensions, or 'None' for a parameter py2fgen passes as a plain array."""

    @property
    def is_host(self) -> bool:
        return self.descriptor.memory_space == py2fgen.MemorySpace.HOST


@dataclasses.dataclass(frozen=True)
class ScalarParam:
    name: str
    descriptor: py2fgen.ScalarParamDescriptor


@dataclasses.dataclass(frozen=True)
class Signature:
    """The parameters of one exported function."""

    name: str
    function: Callable[..., None]
    """The undecorated function ('__wrapped__')."""
    arrays: tuple[ArrayParam, ...]
    scalars: tuple[ScalarParam, ...]


def _field_dims(
    type_hint: Any, descriptor: py2fgen.ArrayParamDescriptor
) -> tuple[gtx.Dimension, ...] | None:
    # the same decision as py2fgen's 'icon4py_export.field_annotation_mapping_hook'
    maybe_gt4py_type = icon4py_export._get_gt4py_type(type_hint)
    if maybe_gt4py_type is None:
        return None
    dims, _ = icon4py_export._parse_type_spec(maybe_gt4py_type[0])
    if len(dims) != descriptor.rank:
        raise ValueError(
            f"Field annotation with {len(dims)} dimensions for a rank-{descriptor.rank} parameter."
        )
    return tuple(dims)


def signature(function_name: str, exported: Any) -> Signature:
    """The parameters of 'exported' (a py2fgen-exported function), from its descriptors."""
    function = exported.__wrapped__
    type_hints = typing.get_type_hints(function, include_extras=True)
    arrays: list[ArrayParam] = []
    scalars: list[ScalarParam] = []
    for name, descriptor in exported.param_descriptors.items():
        if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
            arrays.append(
                ArrayParam(
                    name=name, descriptor=descriptor, dims=_field_dims(type_hints[name], descriptor)
                )
            )
        else:
            scalars.append(ScalarParam(name=name, descriptor=descriptor))
    return Signature(
        name=function_name, function=function, arrays=tuple(arrays), scalars=tuple(scalars)
    )


@dataclasses.dataclass(frozen=True)
class BoundArray:
    """An array argument: ICON's variable (or another buffer) for an array parameter."""

    param: ArrayParam
    variable: Any
    """The 'comin.variable', or a buffer of the plugin's (NumPy or CuPy)."""
    shape: tuple[int, ...]
    """The used extents (py2fgen's shape)."""
    present: bool
    """Presence of an optional argument."""


@dataclasses.dataclass(frozen=True)
class BoundFunction:
    signature: Signature
    arrays: tuple[BoundArray, ...]


def icon_bound_array(variable: Any, param: ArrayParam, icon_name: str | None = None) -> BoundArray:
    """
    ICON's own variable ('icon_name', default: the argument's name) as the array argument
    'param': py2fgen's argument is the first 'rank' extents of ComIn's 5-D shape, so every other
    extent must be 1, the block axis included (one block per patch, nblks == 1). Only valid
    inside a callback that requested it.
    """
    shape = np.asarray(variable).shape  # the host buffer's ('__cuda_array_interface__' too)
    rank = param.descriptor.rank
    if len(shape) != MAX_RANK or any(e != 1 for e in shape[rank:]):
        raise ValueError(
            f"ICON's variable '{icon_name or param.name}' has the shape {shape}: expected {rank}"
            " used extents and 1 elsewhere (one block, nblks == 1), as py2fgen's argument"
            f" '{param.name}'."
        )
    return BoundArray(param=param, variable=variable, shape=shape[:rank], present=True)


def is_on_device(bound: BoundArray, device_xp: ModuleType | None) -> bool:
    return device_xp is not None and not bound.param.is_host


def _null_argument(param: ArrayParam) -> None:
    # py2fgen's rule for a NULL address ('py2fgen._conversion.as_array')
    if param.descriptor.is_optional:
        return None
    raise ValueError(f"'{param.name}': a NULL address for a non-optional argument.")


def _zero_extent_address(bound: BoundArray, device_xp: ModuleType | None) -> int:
    # the adapter builds no buffer on a NULL host address ('comin_var_get_ptr failed')
    try:
        return variable_data_ptr(bound, device_xp)
    except ValueError:
        return 0


def array_argument(bound: BoundArray, device_xp: ModuleType | None) -> Any:
    """
    Build the Python argument py2fgen would pass for one array parameter.

    MAYBE_DEVICE arguments are CuPy views of ICON's device array if 'device_xp' is CuPy, HOST
    arguments are NumPy views; both are sliced from 5 to 'rank' dimensions. A logical argument
    (an INTEGER 0/1 copy in ICON) becomes a bool copy.

    py2fgen passes 'None' for an optional argument whose address is NULL, and a view of the
    given shape otherwise ('py2fgen._conversion.as_array'). The address is the device address
    for MAYBE_DEVICE arguments on the GPU and the host address otherwise, as ComIn hands out
    ICON's variables. So the same rule applies here:
    - an absent optional argument (ICON has no such variable) becomes 'None';
    - a zero-extent array becomes 'None' if its address is NULL (on the GPU the device address
      of a zero-size array is NULL, e.g. for an empty 'zd_*' list on a rank), else an empty
      array of its shape;
    - any other array with a NULL address becomes 'None' if optional; else it is an error.
    """
    param = bound.param
    if not bound.present:
        return None
    xp = device_xp if (device_xp is not None and is_on_device(bound, device_xp)) else np
    dtype = _BUFFER_DTYPE[param.descriptor.dtype]
    rank = param.descriptor.rank
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
    if param.descriptor.dtype == py2fgen.BOOL:
        array = array != 0
    return array if param.dims is None else _views.field_view(array, param.dims)


def variable_data_ptr(bound: BoundArray, device_xp: ModuleType | None) -> int:
    """The address the argument is built from: the device pointer on the GPU, else the host one."""
    if is_on_device(bound, device_xp):
        return int(bound.variable.__cuda_array_interface__["data"][0])
    return _views.data_ptr(np.asarray(bound.variable))
