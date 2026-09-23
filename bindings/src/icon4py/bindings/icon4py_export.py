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


try:
    import cupy as cp  # type: ignore[import-not-found]
except ImportError:
    cp = None


class IconKind(eve.StrEnum):
    """Fortran kind of an ICON actual argument: `REAL(wp)` or `REAL(vp)`."""

    WP = "wp"
    VP = "vp"


class Intent(eve.StrEnum):
    """
    Whether ICON reads an array argument back after the call.

    Only consulted when ICON's dtype and icon4py's differ for that argument, so that it has to
    cross as a converted copy instead of as a view of ICON's memory.
    """

    #: Not classified yet: the argument may cross as a view, never as a copy.
    UNDECLARED = "undeclared"
    #: ICON does not read it back: a copy is converted in and then discarded.
    IN = "in"
    #: ICON reads it back: a copy is converted in, and converted back out after the call.
    INOUT = "inout"


@dataclasses.dataclass(frozen=True)
class Boundary:
    """How a float array crosses the ICON boundary. Attach with `Wp[...]`, `VpInOut[...]` etc."""

    kind: IconKind
    intent: Intent = Intent.UNDECLARED


# `T` is the field icon4py computes with; the prefix is how ICON declares the argument.
type Wp[T] = Annotated[T, Boundary(IconKind.WP)]
type Vp[T] = Annotated[T, Boundary(IconKind.VP)]
type WpIn[T] = Annotated[T, Boundary(IconKind.WP, Intent.IN)]
type VpIn[T] = Annotated[T, Boundary(IconKind.VP, Intent.IN)]
type WpInOut[T] = Annotated[T, Boundary(IconKind.WP, Intent.INOUT)]
type VpInOut[T] = Annotated[T, Boundary(IconKind.VP, Intent.INOUT)]


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


def _synchronize_device() -> None:
    """Wait for all device work, whichever stream GT4Py ran the granule on."""
    assert cp is not None
    cp.cuda.runtime.deviceSynchronize()


def _synchronize_copies() -> None:
    """Wait for this module's own copies, which run on CuPy's current stream."""
    assert cp is not None
    cp.cuda.get_current_stream().synchronize()


def _as_field(
    dims: Sequence[gtx.Dimension], dtype: np.dtype, intent: Intent, icon_dtype: np.dtype
) -> Callable:
    """
    Map an `ArrayInfo` to the field icon4py computes with.

    `icon_dtype` is what the bindings declare ICON passes, `dtype` what icon4py computes in.
    Zero-copy when they agree. Otherwise the field lives in a buffer owned here, refilled from
    ICON's array on every call and, for `Intent.INOUT`, copied back after the call through the
    `writeback` attribute that py2fgen invokes. Both copies cover ICON's whole array -- every
    row up to `nproma`, halo and padding included, and the full vertical extent -- so points the
    granule never writes also come back rounded to `dtype`.
    """

    # maxsize=2 covers double-buffered arguments (the nnow/nnew swap); anything larger lets
    # buffers pile up when ICON passes a freshly allocated array on every call.
    # The dtype checks below run only on a cache miss. That is sound only because each
    # parameter has its own cache and its C type is fixed by the generated cdef: CFFI pointers
    # hash and compare by address alone, so `double*` and `float*` to one address share a key.
    @functools.lru_cache(maxsize=2)
    def cached(array_info: py2fgen.ArrayInfo, ffi: cffi.FFI) -> tuple[gtx.Field | None, Any]:
        """The field handed to icon4py, and ICON's array if that field is a copy of it."""
        arr = py2fgen.as_array(ffi, array_info)
        if arr is None:
            return None, None
        if arr.dtype != icon_dtype:
            raise TypeError(
                f"ICON passes {arr.dtype} for a field the bindings declare as {icon_dtype}: the "
                "library was generated from different signatures, or with a different "
                "ICON4PY_BINDINGS_ICON_PRECISION than the current "
                f"{config.ICON_PRECISION!r}. Regenerate the bindings."
            )
        _, shape, on_gpu, _ = array_info
        domain = gtx_common.domain({d: s for d, s in zip(dims, shape, strict=True)})
        if arr.dtype == dtype:
            return gtx_common._field(arr, domain=domain), None
        if intent is Intent.UNDECLARED:
            raise TypeError(
                f"ICON passes {arr.dtype} for a field on {[d.value for d in dims]} that icon4py "
                f"computes in {dtype}. Converting it needs a copy, so declare whether ICON reads "
                "it back: annotate it `WpIn`/`VpIn` or `WpInOut`/`VpInOut`."
            )
        xp = cp if on_gpu else np
        buffer = xp.empty_like(arr, dtype=dtype)  # order="K" keeps ICON's column-major layout
        return gtx_common._field(buffer, domain=domain), arr

    def impl(array_info: py2fgen.ArrayInfo, *, ffi: cffi.FFI) -> gtx.Field | None:
        field, fortran = cached(array_info, ffi)
        if fortran is not None:
            field.ndarray[...] = fortran
            if array_info[2]:
                _synchronize_copies()  # before GT4Py launches work that reads the buffer
        return field

    if intent is Intent.INOUT:

        def writeback(array_info: py2fgen.ArrayInfo, field: gtx.Field | None, *, ffi: cffi.FFI):
            if field is None:
                return
            _, fortran = cached(array_info, ffi)
            if fortran is None:
                return  # zero-copy: icon4py already wrote into ICON's memory
            if array_info[2]:
                _synchronize_device()  # the granule's last write
            fortran[...] = field.ndarray
            if array_info[2]:
                _synchronize_copies()  # ICON's `!$ACC WAIT` does not cover CuPy's stream

        impl.writeback = writeback  # type: ignore[attr-defined] # py2fgen's writeback protocol

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
    base, boundary = _split_boundary(annotation)
    maybe_gt4py_type = _get_gt4py_type(base)
    if maybe_gt4py_type is None:
        return None
    gt4py_type, _ = maybe_gt4py_type
    dims, dtype = _parse_type_spec(gt4py_type)
    intent = boundary.intent if boundary is not None else Intent.UNDECLARED
    return _as_field(dims, _NUMPY_DTYPE[dtype], intent, _NUMPY_DTYPE[param_descriptor.dtype])


export = py2fgen.export(
    annotation_descriptor_hook=field_annotation_descriptor_hook,
    annotation_mapping_hook=field_annotation_mapping_hook,
)
