# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import types
import typing
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Final, Union

import cffi
import numpy as np
from gt4py import eve, next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.type_system import (
    type_specifications as ts,
    type_translation as gtx_type_translation,
)

from icon4py.bindings import config
from icon4py.tools import py2fgen


class IconKind(eve.StrEnum):
    """Fortran kind of an ICON actual argument: `REAL(wp)` or `REAL(vp)`."""

    WP = "wp"
    VP = "vp"


@dataclasses.dataclass(frozen=True)
class Boundary:
    """How a float array crosses the ICON boundary. Attach with `Wp[...]` or `Vp[...]`."""

    kind: IconKind


# `T` is the field icon4py computes with; the prefix is how ICON declares the argument.
type Wp[T] = Annotated[T, Boundary(IconKind.WP)]
type Vp[T] = Annotated[T, Boundary(IconKind.VP)]


_FLOAT_KINDS: Final = (ts.ScalarKind.FLOAT32, ts.ScalarKind.FLOAT64)

_NUMPY_DTYPE: Final = {
    ts.ScalarKind.BOOL: np.dtype(np.bool_),
    ts.ScalarKind.INT32: np.dtype(np.int32),
    ts.ScalarKind.INT64: np.dtype(np.int64),
    ts.ScalarKind.FLOAT32: np.dtype(np.float32),
    ts.ScalarKind.FLOAT64: np.dtype(np.float64),
}


def icon_scalar_kind(kind: IconKind) -> ts.ScalarKind:
    """
    C kind ICON passes for an argument of Fortran kind `kind`.

    Mirrors `icon/src/shared/mo_kind.f90:60-67`: `wp` is always `dp`; `vp` is `sp` under
    `__MIXED_PRECISION` and `dp` otherwise. ICON's `__SINGLE_PRECISION` is not supported.
    """
    if kind is IconKind.VP and config.ICON_PRECISION == "mixed":
        return ts.ScalarKind.FLOAT32
    return ts.ScalarKind.FLOAT64


def _parse_type_spec(type_spec: ts.TypeSpec) -> tuple[list[gtx.Dimension], ts.ScalarKind]:
    if isinstance(type_spec, ts.ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, ts.FieldType):
        assert isinstance(type_spec.dtype, ts.ScalarType)
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def _is_optional_type_hint(type_hint: Any) -> bool:
    return typing.get_origin(type_hint) in (
        types.UnionType,
        Union,
    ) and types.NoneType in typing.get_args(type_hint)


def _unpack_optional_type_hint(type_hint: Any) -> tuple[Any, bool]:
    if _is_optional_type_hint(type_hint):
        return next(arg for arg in typing.get_args(type_hint) if arg is not types.NoneType), True
    else:
        return type_hint, False


def _expand_type_alias(type_hint: Any) -> Any:
    """Expand an application of a generic PEP 695 alias, e.g. `Vp[fa.CellKField[vpfloat]]`."""
    origin = typing.get_origin(type_hint)
    if isinstance(origin, typing.TypeAliasType):
        return origin.__value__[typing.get_args(type_hint)]
    return type_hint


def _split_boundary(type_hint: Any) -> tuple[Any, Boundary | None]:
    """
    Separate an ICON `Boundary` marker from the type it annotates.

    Accepts both `Vp[X | None]` and `Vp[X] | None`. Type hints without a marker are returned
    unchanged, so that unmarked annotations take exactly the path they took before.
    """
    non_optional, is_optional = _unpack_optional_type_hint(type_hint)
    expanded = _expand_type_alias(non_optional)
    if typing.get_origin(expanded) is Annotated:
        base, *metadata = typing.get_args(expanded)
        boundaries = [m for m in metadata if isinstance(m, Boundary)]
        if len(boundaries) > 1:
            raise TypeError(f"More than one ICON boundary marker on {type_hint}.")
        if boundaries:
            return (base | None if is_optional else base), boundaries[0]
    return type_hint, None


def _get_gt4py_type(type_hint: Any) -> tuple[ts.TypeSpec, bool] | None:
    non_optional_type, is_optional = _unpack_optional_type_hint(type_hint)
    try:
        return gtx_type_translation.from_type_hint(non_optional_type), is_optional
    except ValueError:
        return None


# TODO(egparedes): possibly use `TypeForm` for the annotation parameter,
# once https://peps.python.org/pep-0747/ is approved.
def field_annotation_descriptor_hook(annotation: Any) -> py2fgen.ParamDescriptor | None:
    """
    Translates GT4Py types to 'ParamDescriptor's.

    'gtx.Field' to 'ArrayParamDescriptor' and GT4Py scalars to 'ScalarParamDescriptor'.

    If types cannot be translated to GT4Py types, we delegate to the next mechanism (return 'None').
    """
    base, boundary = _split_boundary(annotation)
    maybe_gt4py_type = _get_gt4py_type(base)
    if maybe_gt4py_type is None:
        if boundary is not None:
            raise TypeError(f"ICON boundary marker on non-GT4Py type {annotation}.")
        return None

    gt4py_type, is_optional = maybe_gt4py_type
    dims, dtype = _parse_type_spec(gt4py_type)
    if not dims:
        if boundary is not None:
            raise TypeError(
                f"ICON boundary marker on scalar {annotation}; scalars cross by value as REAL(wp)."
            )
        if dtype in _FLOAT_KINDS and dtype != ts.ScalarKind.FLOAT64:
            raise TypeError(
                f"Float scalar {annotation} would cross as {dtype}, but ICON passes REAL(wp) "
                "scalars, which are always double: annotate it `gtx.float64`."
            )
        return py2fgen.ScalarParamDescriptor(dtype=dtype)

    # The generated Fortran/C signature must describe what ICON passes, which is fixed by the
    # argument's ICON kind and ICON's precision -- not by the dtype icon4py computes in.
    if dtype in _FLOAT_KINDS:
        if boundary is None:
            raise TypeError(
                f"Float field {annotation} needs its ICON kind: annotate it `Wp[...]` or `Vp[...]`."
            )
        dtype = icon_scalar_kind(boundary.kind)
    elif boundary is not None:
        raise TypeError(f"ICON boundary marker on non-float field {annotation}.")
    return py2fgen.ArrayParamDescriptor(
        rank=len(dims),
        dtype=dtype,
        memory_space=py2fgen.MemorySpace.MAYBE_DEVICE,
        is_optional=is_optional,
    )


def _as_field(dims: Sequence[gtx.Dimension], dtype: np.dtype, icon_dtype: np.dtype) -> Callable:
    """
    Map an `ArrayInfo` to the field icon4py computes with, as a view of ICON's memory.

    `icon_dtype` is what the bindings declare ICON passes, `dtype` what icon4py computes in.
    """

    # in case the cache lookup is still performance relevant, we can replace it by a custom swap cache
    # (only for substitution mode where we know we have exactly 2 entries)
    # or by even marking fields as constant over the whole program run and immediately return on second call
    @functools.cache
    def impl(array_info: py2fgen.ArrayInfo, *, ffi: cffi.FFI) -> gtx.Field | None:
        arr = py2fgen.as_array(ffi, array_info)
        if arr is None:
            return None
        if arr.dtype != icon_dtype:
            raise TypeError(
                f"ICON passes {arr.dtype} for a field the bindings declare as {icon_dtype}: the "
                "library was generated from different signatures, or with a different "
                "ICON4PY_BINDINGS_ICON_PRECISION than the current "
                f"{config.ICON_PRECISION!r}. Regenerate the bindings."
            )
        if arr.dtype != dtype:
            raise TypeError(
                f"ICON passes {arr.dtype} for a field on {[d.value for d in dims]} that icon4py "
                f"computes in {dtype}; casting at the boundary is not implemented."
            )
        _, shape, _, _ = array_info
        domain = {d: s for d, s in zip(dims, shape, strict=True)}
        return gtx_common._field(arr, domain=gtx_common.domain(domain))

    return impl


def field_annotation_mapping_hook(
    annotation: Any, param_descriptor: py2fgen.ParamDescriptor
) -> Callable | None:
    """
    Translates 'ArrayInfo's to 'gtx.Field' if they are annotated with 'gtx.Field'.

    If the type is not a GT4Py type, we delegate to the default mappping (by returning 'None').
    """
    if not isinstance(param_descriptor, py2fgen.ArrayParamDescriptor):
        return None
    base, _ = _split_boundary(annotation)
    maybe_gt4py_type = _get_gt4py_type(base)
    if maybe_gt4py_type is None:
        return None
    gt4py_type, _ = maybe_gt4py_type
    dims, dtype = _parse_type_spec(gt4py_type)
    return _as_field(dims, _NUMPY_DTYPE[dtype], _NUMPY_DTYPE[param_descriptor.dtype])


export = py2fgen.export(
    annotation_descriptor_hook=field_annotation_descriptor_hook,
    annotation_mapping_hook=field_annotation_mapping_hook,
)
