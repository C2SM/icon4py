# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
import types
from typing import Any

import cffi
import numpy as np
import pytest

from icon4py.tools import py2fgen
from icon4py.tools.py2fgen import _conversion, test_utils


try:
    import cupy as cp  # type: ignore[import-not-found]  # optional dependency without stubs
except ImportError:
    cp = None


@pytest.fixture
def ffi() -> cffi.FFI:
    return cffi.FFI()


@pytest.fixture(
    params=[
        np,
        pytest.param(
            cp,
            # TODO(havogt): move to `requires_gpu` pattern?
            marks=pytest.mark.skipif(
                os.getenv("PY2F_GPU_TESTS") is None, reason="GPU tests only run on CI."
            ),
        ),
    ],
    ids=["numpy", "cupy"],
)
def xp(request: pytest.FixtureRequest) -> types.ModuleType:
    return request.param


def _get_ptr(xp: types.ModuleType, arr: Any) -> int:
    if xp == np:
        return arr.ctypes.data
    else:
        return arr.data.ptr


@pytest.mark.parametrize(
    "ctype, rawdata, expected",
    [
        ("double", [1.0, 2.0, 3.0, 4.0], [[1.0, 3.0], [2.0, 4.0]]),
        ("long", [1, 2, 3, 4], [[1, 3], [2, 4]]),
    ],
)
def test_unpack_column_major(
    xp: types.ModuleType, ctype: str, rawdata: list[Any], expected: list[Any], ffi: cffi.FFI
) -> None:
    expected_result = xp.array(expected)

    arr = xp.array(rawdata)
    ptr = ffi.cast(f"{ctype}*", _get_ptr(xp, arr))

    rows, cols = expected_result.shape

    result = _conversion.unpack(xp, ffi, ptr, rows, cols)
    assert isinstance(result, xp.ndarray)
    assert xp.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "ctype, rawdtype, rawdata, expected",
    [
        ("double", np.float64, [1.0, 2.0, 3.0, 4.0], [[1.0, 3.0], [2.0, 4.0]]),
        ("long", np.int64, [1, 2, 3, 4], [[1, 3], [2, 4]]),
        (
            "unsigned char",
            np.bool_,
            [True, False, True, False, False, True],
            [[True, True, False], [False, False, True]],
        ),
    ],
)
def test_as_array(  # noqa: PLR0917 [too-many-positional-arguments]
    xp: types.ModuleType,
    ctype: str,
    rawdtype: type[np.generic],
    rawdata: list[Any],
    expected: list[Any],
    ffi: cffi.FFI,
) -> None:
    expected_result = xp.array(expected)

    arr = xp.array(rawdata, dtype=rawdtype)
    ptr = ffi.cast(f"{ctype}*", _get_ptr(xp, arr))
    array_info = test_utils.array_info(
        ptr, shape=expected_result.shape, on_gpu=xp != np, is_optional=False
    )

    result = _conversion.as_array(ffi, array_info)

    assert isinstance(result, xp.ndarray)
    assert xp.array_equal(result, expected_result)


def test_default_mapping_hook_array(ffi: cffi.FFI) -> None:
    array_ptr = ffi.new("int[10]")

    array_mapper = _conversion.default_mapping(
        None,
        py2fgen.ArrayParamDescriptor(
            rank=1, dtype=py2fgen.INT32, memory_space=py2fgen.MemorySpace.HOST, is_optional=False
        ),
    )
    assert array_mapper is not None
    result = array_mapper(
        test_utils.array_info(ptr=array_ptr, shape=(10,), on_gpu=False, is_optional=False),
        ffi=ffi,
    )

    assert isinstance(result, np.ndarray)


def test_default_mapping_hook_bool(ffi: cffi.FFI) -> None:
    # Scalar BOOL is `unsigned char` on the C side, so CFFI delivers a Python `int`
    # which the mapper converts to `bool`.
    mapper = _conversion.default_mapping(None, py2fgen.ScalarParamDescriptor(dtype=py2fgen.BOOL))
    assert mapper is not None
    assert mapper(0, ffi=ffi) is False
    assert mapper(1, ffi=ffi) is True
    # NVHPC writes `.TRUE.` as 0xFF; the mapper must still produce `True`.
    assert mapper(255, ffi=ffi) is True
