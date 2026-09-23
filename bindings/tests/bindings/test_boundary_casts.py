# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Arguments whose ICON dtype differs from icon4py's cross as converted copies.

Here ICON passes double and icon4py computes in single. Both are spelled out -- `Wp` arguments,
which ICON passes as double under every ICON precision, and `gtx.float32` -- so that the tests
depend on neither precision setting. None of them needs serialized data.
"""

import cffi
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.bindings import icon4py_export
from icon4py.bindings.icon4py_export import WpIn, WpInOut
from icon4py.tools.py2fgen import test_utils


SomeDim = gtx.Dimension("SomeDim")
LevelDim = gtx.Dimension("LevelDim", kind=gtx.DimensionKind.VERTICAL)

pytestmark = pytest.mark.single_precision_ready

seen: list[np.ndarray] = []


@icon4py_export.export
def scale_inout(field: WpInOut[gtx.Field[gtx.Dims[SomeDim], gtx.float32]], factor: gtx.float64):
    field.ndarray[...] *= np.float32(factor)


@icon4py_export.export
def scale_in(field: WpIn[gtx.Field[gtx.Dims[SomeDim], gtx.float32]], factor: gtx.float64):
    field.ndarray[...] *= np.float32(factor)  # must not reach ICON


@icon4py_export.export
def record(field: WpIn[gtx.Field[gtx.Dims[SomeDim], gtx.float32]]):
    seen.append(field.asnumpy().copy())


@icon4py_export.export
def scale_inout_double(
    field: WpInOut[gtx.Field[gtx.Dims[SomeDim], gtx.float64]], factor: gtx.float64
):
    seen.append(field.ndarray)
    field.ndarray[...] *= factor


@icon4py_export.export
def record_2d(field: WpIn[gtx.Field[gtx.Dims[SomeDim, LevelDim], gtx.float32]]):
    seen.append(field.ndarray)


@icon4py_export.export
def scale_optional_inout(
    field: WpInOut[gtx.Field[gtx.Dims[SomeDim], gtx.float32] | None], factor: gtx.float64
):
    seen.append(field)
    if field is not None:
        field.ndarray[...] *= np.float32(factor)


@pytest.fixture(autouse=True)
def _clear_seen():
    seen.clear()


def _values(n: int = 7) -> np.ndarray:
    # not exactly representable in single, so a missing conversion shows up as a mismatch
    return np.linspace(0.1, 1.3, n, dtype=np.float64)


def test_inout_is_converted_in_and_back_out():
    ffi = cffi.FFI()
    fortran = _values()
    expected = (fortran.astype(np.float32) * np.float32(3.0)).astype(np.float64)

    scale_inout(
        ffi=ffi, perf_counters=None, field=test_utils.array_to_array_info(fortran), factor=3.0
    )

    assert fortran.dtype == np.float64
    np.testing.assert_array_equal(fortran, expected)


def test_in_is_not_written_back():
    """A pure input must not come back truncated to single precision."""
    ffi = cffi.FFI()
    fortran = _values()
    before = fortran.copy()

    scale_in(ffi=ffi, perf_counters=None, field=test_utils.array_to_array_info(fortran), factor=3.0)

    np.testing.assert_array_equal(fortran, before)


def test_copy_is_refreshed_on_every_call():
    """The same pointer on the next call must deliver ICON's new values, not a cached copy."""
    ffi = cffi.FFI()
    fortran = _values()
    array_info = test_utils.array_to_array_info(fortran)

    record(ffi=ffi, perf_counters=None, field=array_info)
    fortran[...] = 2.0 * fortran
    record(ffi=ffi, perf_counters=None, field=array_info)

    assert seen[0].dtype == np.float32
    np.testing.assert_array_equal(seen[0], _values().astype(np.float32))
    np.testing.assert_array_equal(seen[1], (2.0 * _values()).astype(np.float32))


def test_undeclared_intent_refuses_to_copy_and_names_the_argument():
    ffi = cffi.FFI()

    @icon4py_export.export
    def undeclared(field: icon4py_export.Wp[gtx.Field[gtx.Dims[SomeDim], gtx.float32]]):
        pass

    with pytest.raises(TypeError, match="declare whether ICON reads it back") as excinfo:
        undeclared(ffi=ffi, perf_counters=None, field=test_utils.array_to_array_info(_values()))

    assert "while converting argument 'field' of 'undeclared'" in excinfo.value.__notes__


def test_inout_without_conversion_stays_zero_copy():
    ffi = cffi.FFI()
    fortran = _values()

    scale_inout_double(
        ffi=ffi, perf_counters=None, field=test_utils.array_to_array_info(fortran), factor=3.0
    )

    assert np.shares_memory(seen[0], fortran)
    np.testing.assert_array_equal(fortran, 3.0 * _values())


def test_buffers_are_reused_for_double_buffered_arguments():
    """nnow/nnew alternate between two ICON arrays; that must not allocate on every call."""
    ffi = cffi.FFI()
    a, b, c = _values(), 2.0 * _values(), 3.0 * _values()
    info_a, info_b, info_c = (test_utils.array_to_array_info(x) for x in (a, b, c))

    @icon4py_export.export
    def keep(field: WpIn[gtx.Field[gtx.Dims[SomeDim], gtx.float32]]):
        seen.append(field)

    for info in (info_a, info_b, info_a, info_b, info_c):
        keep(ffi=ffi, perf_counters=None, field=info)

    assert seen[2] is seen[0] and seen[3] is seen[1]
    np.testing.assert_array_equal(seen[4].asnumpy(), c.astype(np.float32))  # after eviction


def test_copy_keeps_icons_column_major_layout():
    ffi = cffi.FFI()
    fortran = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))

    record_2d(ffi=ffi, perf_counters=None, field=test_utils.array_to_array_info(fortran))

    assert seen[0].dtype == np.float32
    assert seen[0].flags.f_contiguous
    np.testing.assert_array_equal(seen[0], fortran.astype(np.float32))


def test_optional_inout_accepts_an_unassociated_pointer():
    ffi = cffi.FFI()
    null = test_utils.array_info(ffi.NULL, (7,), False, True)

    scale_optional_inout(ffi=ffi, perf_counters=None, field=null, factor=3.0)

    assert seen == [None]
