# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Write-through tests of 'field_view', the zero-copy helper of the icon4py ComIn plugin.

The buffers are shaped like ComIn variables: 5-D, Fortran order, trailing extents 1, seen
through the buffer protocol (NumPy) or '__cuda_array_interface__' (CuPy) and sliced to rank 2.
"""

import numpy as np
import pytest
from gt4py import next as gtx
from gt4py.next import common as gtx_common

from icon4py.bindings.comin import _views
from icon4py.model.common import dimension as dims, field_type_aliases as fa


NPROMA, NLEV = 7, 5
CELL_K = (dims.CellDim, dims.KDim)


@gtx.field_operator
def _add_one(a: fa.CellKField[gtx.float64]) -> fa.CellKField[gtx.float64]:
    return a + 1.0


@gtx.program
def add_one(a: fa.CellKField[gtx.float64], out: fa.CellKField[gtx.float64]):
    _add_one(a, out=out)


def comin_buffer(xp, fill: float = 0.0):
    return xp.full((NPROMA, NLEV, 1, 1, 1), fill, dtype=np.float64, order="F")


def numpy_comin_view(buffer: np.ndarray) -> np.ndarray:
    # what 'np.asarray(comin.variable)' gives: a PEP 3118 memoryview over ICON's memory
    return np.asarray(memoryview(buffer))[:, :, 0, 0, 0]


class _CudaArrayInterface:
    def __init__(self, array):
        self.__cuda_array_interface__ = array.__cuda_array_interface__


def cupy_comin_view(cp, buffer):
    # what 'cupy.asarray(comin.variable)' gives: a view through '__cuda_array_interface__'
    return cp.asarray(_CudaArrayInterface(buffer))[:, :, 0, 0, 0]


def cupy_or_skip():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No GPU.")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("No GPU.")
    return cp


def test_field_view_aliases_the_buffer():
    buffer = comin_buffer(np)
    view = numpy_comin_view(buffer)
    field = _views.field_view(view, CELL_K)

    assert _views.data_ptr(field.ndarray) == _views.data_ptr(view) == buffer.ctypes.data
    assert field.ndarray.strides == view.strides == (8, 8 * NPROMA)  # py2fgen's layout
    assert np.shares_memory(field.ndarray, buffer)
    assert field.domain == gtx_common.domain({dims.CellDim: NPROMA, dims.KDim: NLEV})
    assert all(r.start == 0 for r in field.domain.ranges)


def test_field_view_writes_through_ndarray():
    buffer = comin_buffer(np)
    field = _views.field_view(numpy_comin_view(buffer), CELL_K)
    field.ndarray[...] = 3.0
    assert np.all(buffer == 3.0)


@pytest.mark.parametrize("program_backend", [None, gtx.gtfn_cpu], ids=["embedded", "gtfn_cpu"])
def test_field_view_writes_through_program(program_backend):
    source = comin_buffer(np)
    source[..., 0, 0, 0] = np.arange(NPROMA * NLEV, dtype=np.float64).reshape(NPROMA, NLEV)
    target = comin_buffer(np, fill=-1.0)
    a = _views.field_view(numpy_comin_view(source), CELL_K)
    out = _views.field_view(numpy_comin_view(target), CELL_K)

    add_one.with_backend(program_backend)(a, out, offset_provider={})

    np.testing.assert_array_equal(target, source + 1.0)


@pytest.mark.parametrize("program_backend", [None, gtx.gtfn_cpu], ids=["embedded", "gtfn_cpu"])
def test_field_view_writes_through_program_in_place(program_backend):
    buffer = comin_buffer(np, fill=2.0)
    field = _views.field_view(numpy_comin_view(buffer), CELL_K)

    add_one.with_backend(program_backend)(field, field, offset_provider={})

    assert np.all(buffer == 3.0)


def test_as_field_copies():
    # why the plugin needs 'field_view': 'gtx.as_field' copies, so writes never reach ICON
    buffer = comin_buffer(np)
    field = gtx.as_field(list(CELL_K), numpy_comin_view(buffer))
    field.ndarray[...] = 1.0
    assert not np.shares_memory(field.ndarray, buffer)
    assert np.all(buffer == 0.0)


@pytest.mark.parametrize("copy", [np.asfortranarray, np.ascontiguousarray], ids=["F", "C"])
def test_field_view_raises_if_field_copies(monkeypatch, copy):
    # '_field' never copies an ndarray today; simulate a GT4Py that would
    original = gtx_common._field

    def copying_field(data, **kwargs):
        return original(copy(data.copy()), **kwargs)

    monkeypatch.setattr(gtx_common, "_field", copying_field)
    with pytest.raises(ValueError, match="does not alias"):
        _views.field_view(numpy_comin_view(comin_buffer(np)), CELL_K)


def test_field_view_rejects_wrong_rank():
    with pytest.raises(ValueError):
        _views.field_view(numpy_comin_view(comin_buffer(np)), (dims.CellDim,))


def test_field_view_cupy_aliases_and_writes_through():
    cp = cupy_or_skip()
    buffer = comin_buffer(cp)
    view = cupy_comin_view(cp, buffer)
    field = _views.field_view(view, CELL_K)

    assert _views.data_ptr(field.ndarray) == buffer.data.ptr
    assert field.ndarray.strides == view.strides == (8, 8 * NPROMA)
    field.ndarray[...] = 3.0
    add_one(field, field, offset_provider={})  # embedded
    cp.cuda.runtime.deviceSynchronize()
    assert bool(cp.all(buffer == 4.0))
