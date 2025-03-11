# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import typing
from collections.abc import Sequence
from types import NoneType
from typing import Any, Callable, Optional, Union

import cffi
from gt4py import next as gtx
from gt4py.next import common as gtx_common
from gt4py.next.type_system import (
    type_specifications as ts,
    type_translation as gtx_type_translation,
)

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _template, wrapper_utils


def _parse_type_spec(type_spec: ts.TypeSpec) -> tuple[list[gtx.Dimension], ts.ScalarKind]:
    if isinstance(type_spec, ts.ScalarType):
        return [], type_spec.kind
    elif isinstance(type_spec, ts.FieldType):
        return type_spec.dims, type_spec.dtype.kind
    else:
        raise ValueError(f"Unsupported type specification: {type_spec}")


def _is_optional_type_hint(type_hint: Any) -> bool:
    return typing.get_origin(type_hint) is Union and typing.get_args(type_hint)[1] is NoneType


def _unpack_optional_type_hint(type_hint: Any) -> tuple[Any, bool]:
    if _is_optional_type_hint(type_hint):
        return typing.get_args(type_hint)[0], True
    else:
        return type_hint, False


def _get_gt4py_type(type_hint: Any) -> Optional[tuple[ts.TypeSpec, bool]]:
    non_optional_type, is_optional = _unpack_optional_type_hint(type_hint)
    try:
        return gtx_type_translation.from_type_hint(non_optional_type), is_optional
    except ValueError:
        return None


def field_annotation_descriptor_hook(annotation: Any) -> Optional[py2fgen.ParamDescriptor]:
    maybe_gt4py_type = _get_gt4py_type(annotation)
    if maybe_gt4py_type is None:
        return None

    gt4py_type, is_optional = maybe_gt4py_type
    dims, dtype = _parse_type_spec(gt4py_type)
    if len(dims) > 0:
        return py2fgen.ArrayParamDescriptor(
            rank=len(dims),
            dtype=dtype,
            device=_template.DeviceType.MAYBE_DEVICE,
            is_optional=is_optional,
        )
    else:
        return py2fgen.ScalarParamDescriptor(dtype=dtype)


def _as_field(dims: Sequence[gtx.Dimension], scalar_kind: ts.ScalarKind) -> Callable:
    # in case the cache lookup is still performance relevant, we can replace it by a custom swap cache
    # (only for substitution mode where we know we have exactly 2 entries)
    # or by even marking fields as constant over the whole program run and immediately return on second call
    @functools.lru_cache(maxsize=None)
    def impl(
        array_descriptor: wrapper_utils.ArrayDescriptor, *, ffi: cffi.FFI
    ) -> Optional[gtx.Field]:
        arr = wrapper_utils.as_array(ffi, array_descriptor, scalar_kind)
        if arr is None:
            return None
        domain = {d: s for d, s in zip(dims, array_descriptor[1], strict=True)}
        return gtx_common._field(arr, domain=gtx_common.domain(domain))

    return impl


def field_annotation_mapping_hook(
    annotation: Any, param_descriptor: py2fgen.ParamDescriptor
) -> Callable:
    if not isinstance(param_descriptor, py2fgen.ArrayParamDescriptor):
        return None
    maybe_gt4py_type = _get_gt4py_type(annotation)
    if maybe_gt4py_type is None:
        return None
    else:
        gt4py_type, is_optional = maybe_gt4py_type
        dims, dtype = _parse_type_spec(gt4py_type)
        return _as_field(dims, dtype)


export = py2fgen.export(
    annotation_descriptor_hook=field_annotation_descriptor_hook,
    annotation_mapping_hook=field_annotation_mapping_hook,
)
