# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import cffi
from gt4py import next as gtx
from gt4py.next.type_system import type_specifications as ts

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import wrapper_utils
from icon4py.tools.py2fgen.wrappers import icon4py_export


export_with_mapping_hook = py2fgen.export(
    annotation_mapping_hook=icon4py_export.field_annotation_mapping_hook,
    param_descriptors={
        "a": py2fgen.ArrayParamDescriptor(
            rank=1, dtype=ts.ScalarKind.INT32, device=py2fgen.DeviceType.HOST, is_optional=False
        ),
        "b": py2fgen.ScalarParamDescriptor(dtype=ts.ScalarKind.INT32),
    },
)

SomeDim = gtx.Dimension("SomeDim")


def make_array_descriptor(
    ptr: cffi.FFI.CData,
    shape: tuple[int, ...],
    on_gpu: bool,
    is_optional: bool,
) -> wrapper_utils.ArrayDescriptor:
    return (ptr, shape, on_gpu, is_optional)


@export_with_mapping_hook
def foo(a: gtx.Field[gtx.Dims[SomeDim], gtx.int32], b: gtx.int32):
    return a, b


def test_mapping_hook():
    ffi = cffi.FFI()

    array_ptr = ffi.new("int[10]")

    result_a, result_b = foo(
        ffi=ffi,
        meta={},
        a=make_array_descriptor(shape=(10,), ptr=array_ptr, on_gpu=False, is_optional=False),
        b=5,
    )
    assert hasattr(result_a, "ndarray")
    assert result_b == 5
