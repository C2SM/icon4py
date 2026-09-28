# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Marshalling between ICON's ComIn argument variables and py2fgen-exported functions.

ICON's ComIn backend exposes, on domain 1,
- every array argument 'arg' of an exported function 'fn' as the variable 'icon4py_<fn>_<arg>',
  bound to exactly the array py2fgen would get, with the metadata 'datatype', 'icon4py_rank',
  'icon4py_shape' (padded to 5 dimensions) and 'icon4py_present';
- every scalar argument as the metadata key 'icon4py_<arg>' of the host carrier variable
  'icon4py_<fn>_ctl' (one INTEGER element, through which the plugin acknowledges a call).

This module derives the expected variables from the function's 'param_descriptors', checks
ICON's metadata against them in the secondary constructor, and builds the keyword arguments
of the undecorated function with py2fgen's layout: the same shapes, dtypes and Fortran-order
strides, domains starting at 0, 'gtx.Field's only for Field-annotated parameters.
"""

import dataclasses
import typing
from collections.abc import Callable, Collection, Mapping, Sequence
from types import ModuleType
from typing import Any, Final

import numpy as np
from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as ts

from icon4py.bindings import icon4py_export
from icon4py.bindings.comin import _views
from icon4py.tools import py2fgen


DOMAIN_ID: Final = 1
"""icon4py runs on one domain; all its ComIn variables live on domain 1."""

MAX_RANK: Final = 5
"""ComIn variables are 5-dimensional; ICON pads the unused trailing extents with 1."""

DATATYPE_KEY: Final = "datatype"
RANK_KEY: Final = "icon4py_rank"
SHAPE_KEY: Final = "icon4py_shape"
PRESENT_KEY: Final = "icon4py_present"
PASS_KEY: Final = "icon4py_pass"
"""Carrier metadata of the init functions: the number of the current 'perform_nh_stepping' pass."""
CALL_COUNT_KEY: Final = "icon4py_call_count"
"""Carrier metadata of a run function: the number of the current call."""

_COMIN_DATATYPE: Final[Mapping[ts.ScalarKind, str]] = {
    py2fgen.FLOAT64: "COMIN_VAR_DATATYPE_DOUBLE",
    py2fgen.FLOAT32: "COMIN_VAR_DATATYPE_FLOAT",
    py2fgen.INT32: "COMIN_VAR_DATATYPE_INT",
    py2fgen.BOOL: "COMIN_VAR_DATATYPE_INT",  # ComIn has no logical type: ICON exposes 0/1 copies
}
_BUFFER_DTYPE: Final[Mapping[ts.ScalarKind, np.dtype]] = {
    py2fgen.FLOAT64: np.dtype(np.float64),
    py2fgen.FLOAT32: np.dtype(np.float32),
    py2fgen.INT32: np.dtype(np.int32),
    py2fgen.BOOL: np.dtype(np.int32),
}


def variable_name(function_name: str, param_name: str) -> str:
    return f"icon4py_{function_name}_{param_name}"


def carrier_name(function_name: str) -> str:
    return f"icon4py_{function_name}_ctl"


def scalar_key(param_name: str) -> str:
    return f"icon4py_{param_name}"


@dataclasses.dataclass(frozen=True)
class ArrayParam:
    name: str
    variable: str
    descriptor: py2fgen.ArrayParamDescriptor
    dims: tuple[gtx.Dimension, ...] | None
    """The Field dimensions, or 'None' for a parameter py2fgen passes as a plain array."""

    @property
    def is_host(self) -> bool:
        return self.descriptor.memory_space == py2fgen.MemorySpace.HOST


@dataclasses.dataclass(frozen=True)
class ScalarParam:
    name: str
    key: str
    descriptor: py2fgen.ScalarParamDescriptor


@dataclasses.dataclass(frozen=True)
class Signature:
    """The ComIn view of one exported function."""

    name: str
    function: Callable[..., None]
    """The undecorated function ('__wrapped__')."""
    carrier: str
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
    """Derive the ComIn names of 'exported' (a py2fgen-exported function) from its descriptors."""
    function = exported.__wrapped__
    type_hints = typing.get_type_hints(function, include_extras=True)
    arrays: list[ArrayParam] = []
    scalars: list[ScalarParam] = []
    for name, descriptor in exported.param_descriptors.items():
        if isinstance(descriptor, py2fgen.ArrayParamDescriptor):
            arrays.append(
                ArrayParam(
                    name=name,
                    variable=variable_name(function_name, name),
                    descriptor=descriptor,
                    dims=_field_dims(type_hints[name], descriptor),
                )
            )
        else:
            scalars.append(ScalarParam(name=name, key=scalar_key(name), descriptor=descriptor))
    return Signature(
        name=function_name,
        function=function,
        carrier=carrier_name(function_name),
        arrays=tuple(arrays),
        scalars=tuple(scalars),
    )


def scalar_value(param: ScalarParam, raw: Any) -> bool | int | float:
    """Convert carrier metadata to the Python scalar py2fgen passes; strict about the type."""
    kind = param.descriptor.dtype
    if kind == py2fgen.BOOL:
        if isinstance(raw, int) and raw in (0, 1):  # ICON stores a logical as 0/1
            return bool(raw)
    elif kind in (py2fgen.INT32, py2fgen.INT64):
        if isinstance(raw, int) and not isinstance(raw, bool):
            return int(raw)
    elif kind in (py2fgen.FLOAT32, py2fgen.FLOAT64):
        if isinstance(raw, float):
            return float(raw)
    raise TypeError(f"Metadata '{param.key}': expected a {kind.name} value, got {raw!r}.")


@dataclasses.dataclass(frozen=True)
class BoundArray:
    param: ArrayParam
    variable: Any
    """The 'comin.variable'."""
    shape: tuple[int, ...]
    """The used extents, 'icon4py_shape[:rank]'; fixed when ICON exposed the variable."""
    present: bool
    """Presence of an optional argument; fixed when ICON exposed the variable."""


@dataclasses.dataclass(frozen=True)
class BoundFunction:
    signature: Signature
    arrays: tuple[BoundArray, ...]
    carrier: Any


_MISSING: Final = object()


def _get(metadata: Mapping[str, Any], key: str) -> Any:
    return metadata.get(key, _MISSING)


def _show(value: Any) -> str:
    return "missing" if value is _MISSING else repr(value)


def check_array_metadata(
    comin: ModuleType, param: ArrayParam, metadata: Mapping[str, Any]
) -> tuple[tuple[int, ...], bool, list[str]]:
    """Check ICON's metadata of one argument variable; return (used extents, presence, errors)."""
    errors: list[str] = []
    rank = param.descriptor.rank
    where = f"'{param.variable}'"

    datatype_name = _COMIN_DATATYPE.get(param.descriptor.dtype)
    datatype = _get(metadata, DATATYPE_KEY)
    if datatype_name is None:
        errors.append(f"{where}: dtype {param.descriptor.dtype.name} has no ComIn datatype.")
    elif datatype != getattr(comin, datatype_name):
        errors.append(f"{where}: '{DATATYPE_KEY}' is {_show(datatype)}, expected {datatype_name}.")

    icon_rank = _get(metadata, RANK_KEY)
    if icon_rank != rank:
        errors.append(f"{where}: '{RANK_KEY}' is {_show(icon_rank)}, expected {rank}.")

    present_raw = _get(metadata, PRESENT_KEY)
    present = present_raw in (0, 1) and bool(present_raw)
    if present_raw not in (0, 1):
        errors.append(f"{where}: '{PRESENT_KEY}' is {_show(present_raw)}, expected 0 or 1.")
    elif not present and not param.descriptor.is_optional:
        errors.append(f"{where}: the argument is not optional, but ICON marks it absent.")

    shape5 = _get(metadata, SHAPE_KEY)
    shape: tuple[int, ...] = ()
    if not present:
        pass  # the plugin never touches an absent argument, whatever its exposed shape
    elif not (
        isinstance(shape5, list)
        and len(shape5) == MAX_RANK
        and all(isinstance(s, int) and s >= 0 for s in shape5)
    ):
        errors.append(f"{where}: '{SHAPE_KEY}' is {_show(shape5)}, expected 5 extents.")
    else:
        shape = tuple(shape5[:rank])
        if any(s != 1 for s in shape5[rank:]):
            errors.append(
                f"{where}: '{SHAPE_KEY}' {shape5} pads a rank-{rank} array with extents != 1."
            )
    return shape, present, errors


def check_carrier_metadata(
    comin: ModuleType, signature: Signature, metadata: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    where = f"'{signature.carrier}'"
    datatype = _get(metadata, DATATYPE_KEY)
    if datatype != comin.COMIN_VAR_DATATYPE_INT:
        errors.append(
            f"{where}: '{DATATYPE_KEY}' is {_show(datatype)}, expected COMIN_VAR_DATATYPE_INT."
        )
    shape5 = _get(metadata, SHAPE_KEY)
    if shape5 != [1] * MAX_RANK:
        errors.append(f"{where}: '{SHAPE_KEY}' is {_show(shape5)}, expected {[1] * MAX_RANK}.")
    for param in signature.scalars:
        raw = _get(metadata, param.key)
        if raw is _MISSING:
            errors.append(f"{where}: scalar '{param.name}' is missing (metadata '{param.key}').")
            continue
        try:
            scalar_value(param, raw)
        except TypeError as error:
            errors.append(f"{where}: {error}")
    return errors


def bind(
    comin: ModuleType,
    signature: Signature,
    *,
    context: Sequence[Any],
    flags: int,
    carrier_flags: int,
    exposed: Collection[tuple[str, int]],
    errors: list[str],
) -> BoundFunction:
    """
    Request ('comin.var_get') every argument variable and the carrier of one function.

    Only valid in ComIn's secondary constructor. Problems are appended to 'errors', so that the
    caller can report all of them at once.
    """
    arrays: list[BoundArray] = []
    for param in signature.arrays:
        descriptor = (param.variable, DOMAIN_ID)
        if descriptor not in exposed:
            errors.append(f"'{param.variable}': ICON does not expose this variable.")
            continue
        shape, present, array_errors = check_array_metadata(
            comin, param, comin.metadata(descriptor)
        )
        errors.extend(array_errors)
        try:
            variable = comin.var_get(list(context), descriptor, flags)
        except Exception as error:
            errors.append(f"'{param.variable}': 'var_get' failed: {error}")
            continue
        arrays.append(BoundArray(param=param, variable=variable, shape=shape, present=present))

    carrier = None
    descriptor = (signature.carrier, DOMAIN_ID)
    if descriptor not in exposed:
        errors.append(f"'{signature.carrier}': ICON does not expose this carrier variable.")
    else:
        errors.extend(check_carrier_metadata(comin, signature, comin.metadata(descriptor)))
        try:
            carrier = comin.var_get(list(context), descriptor, carrier_flags)
        except Exception as error:
            errors.append(f"'{signature.carrier}': 'var_get' failed: {error}")
    return BoundFunction(signature=signature, arrays=tuple(arrays), carrier=carrier)


def is_on_device(bound: BoundArray, device_xp: ModuleType | None) -> bool:
    return device_xp is not None and not bound.param.is_host


def _null_argument(param: ArrayParam) -> None:
    # py2fgen's rule for a NULL address ('py2fgen._conversion.as_array')
    if param.descriptor.is_optional:
        return None
    raise ValueError(f"'{param.variable}': ICON bound a NULL pointer to a non-optional argument.")


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
    for MAYBE_DEVICE arguments on the GPU and the host address otherwise, and ICON's ComIn
    backend exposes exactly those addresses. So the same rule applies here:
    - an absent optional (ICON exposes NULL addresses) becomes 'None', without touching the
      variable;
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
                f"'{param.variable}': the view has shape {array.shape} and dtype {array.dtype},"
                f" expected {bound.shape} and {dtype}."
            )
    if param.descriptor.dtype == py2fgen.BOOL:
        array = array != 0
    return array if param.dims is None else _views.field_view(array, param.dims)


def carrier_metadata(comin: ModuleType, bound: BoundFunction) -> Mapping[str, Any]:
    return comin.metadata((bound.signature.carrier, DOMAIN_ID))


def arguments(
    comin: ModuleType, bound: BoundFunction, device_xp: ModuleType | None
) -> dict[str, Any]:
    """Build all keyword arguments of 'bound.signature.function' for the current call."""
    kwargs = {b.param.name: array_argument(b, device_xp) for b in bound.arrays}
    metadata = carrier_metadata(comin, bound)
    for param in bound.signature.scalars:
        kwargs[param.name] = scalar_value(param, metadata[param.key])
    return kwargs


def write_carrier(bound: BoundFunction, value: int) -> None:
    """Write 'value' into the single element of the carrier (host memory)."""
    element = np.asarray(bound.carrier)
    if element.size != 1 or element.dtype != np.int32:
        raise ValueError(
            f"'{bound.signature.carrier}': expected one int32 element, got shape"
            f" {element.shape} and dtype {element.dtype}."
        )
    element[(0,) * element.ndim] = value


def variable_data_ptr(bound: BoundArray, device_xp: ModuleType | None) -> int:
    """The address ComIn hands out for a variable: device pointer on the GPU, else host pointer."""
    if is_on_device(bound, device_xp):
        return int(bound.variable.__cuda_array_interface__["data"][0])
    return _views.data_ptr(np.asarray(bound.variable))
