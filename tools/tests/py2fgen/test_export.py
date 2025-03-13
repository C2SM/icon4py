# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import cffi
import numpy as np

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _export


def make_array_descriptor(
    ptr: "cffi.FFI.CData",  # TODO don't `from __future__ import annotations` otherwise the gt4py annotation will be a string
    shape: tuple[int, ...],
    on_gpu: bool,
    is_optional: bool,
) -> py2fgen.ArrayDescriptor:
    return (ptr, shape, on_gpu, is_optional)


def test_default_mapping_hook_array():
    ffi = cffi.FFI()

    array_ptr = ffi.new("int[10]")

    array_mapper = _export.default_mapping(
        None,
        py2fgen.ArrayParamDescriptor(
            rank=1, dtype=py2fgen.INT32, device=py2fgen.DeviceType.HOST, is_optional=False
        ),
    )
    result = array_mapper(
        make_array_descriptor(ptr=array_ptr, shape=(10,), on_gpu=False, is_optional=False), ffi=ffi
    )

    assert isinstance(result, np.ndarray)


_ffi_not_needed = None


def test_default_mapping_hook_bool():
    bool_mapper = _export.default_mapping(None, py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL))

    true_result = bool_mapper(-1, ffi=_ffi_not_needed)
    assert isinstance(true_result, bool)
    assert true_result
    false_result = bool_mapper(0, ffi=_ffi_not_needed)
    assert isinstance(false_result, bool)
    assert not false_result
