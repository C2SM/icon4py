# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests of the comparisons of '_compare.py', with which the py2fgen probe compares the plugin's
inputs and objects with py2fgen's ('test_comin_probe.py').
"""

import math
import struct

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.bindings.comin import _compare
from icon4py.model.common import dimension as dims


# ---- comparisons ------------------------------------------------------------------------------


def test_identical_with_padding():
    new = np.arange(24, dtype=np.int32).reshape(8, 3)
    result = _compare.compare_arrays(new, new.copy(order="F"), real=5)
    assert result.identical
    assert result.details == "int32 (8, 3); real 15: 0 differ; padding 9: 0 differ"


def test_differing_real_entry():
    new = np.linspace(1.0, 2.0, 12).reshape(6, 2)
    old = new.copy()
    old[3, 1] = np.nextafter(old[3, 1], np.inf)
    old[4, 0] = old[4, 0] * 2
    result = _compare.compare_arrays(new, old, real=5)
    assert not result.identical and (result.real_differ, result.padding_differ) == (2, 0)
    assert "real 10: 2 differ, first (3, 1), max abs" in result.details
    one_ulp = _compare.compare_arrays(new[3:4, 1], old[3:4, 1])
    assert one_ulp.details.endswith("max ulp 1; padding 0: 0 differ")


def test_padding_only_difference_is_identical():
    new = np.zeros(10, dtype=np.float64)
    old = new.copy()
    old[7:] = np.inf  # e.g. 1/0 beyond the local edges
    result = _compare.compare_arrays(new, old, real=7)
    assert result.identical and result.padding_differ == 3
    assert result.details == "float64 (10,); real 7: 0 differ; padding 3: 3 differ"
    assert not _compare.compare_arrays(new, old).identical  # no padding: all entries are real


def test_padding_along_another_axis():
    new = np.zeros((6, 2, 5))
    old = new.copy()
    old[:, :, 4] = 1.0
    assert _compare.compare_arrays(new, old, real=4, axis=2).identical
    assert not _compare.compare_arrays(new, old, real=5, axis=2).identical


def test_zero_entries_are_counted():
    """A difference against a zero (e.g. ICON's 0 where it does not compute a field) is named."""
    old = np.array([0.0, 2.0, 0.0, 4.0])
    result = _compare.compare_arrays(np.array([1.0, 2.0, 0.5, 0.0]), old)
    assert result.details.endswith(
        "real 4: 3 differ, first (0,), max abs 4, max rel inf, max ulp 4616189618054758400,"
        " reference zero at 2, new route zero at 1; padding 0: 0 differ"
    )


def test_nan_payload_and_signed_zero():
    nan_a = struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0]
    nan_b = struct.unpack("<d", bytes.fromhex("020000000000f87f"))[0]
    new = np.array([nan_a, 0.0, 1.0])
    same_nan = np.array([nan_a, 0.0, 1.0])
    assert _compare.compare_arrays(new, same_nan).identical  # NaN == NaN when the bits are equal
    other_nan = np.array([nan_b, 0.0, 1.0])
    assert not _compare.compare_arrays(new, other_nan).identical
    negative_zero = np.array([nan_a, -0.0, 1.0])
    result = _compare.compare_arrays(new, negative_zero)
    assert not result.identical and "first (1,), max abs 0" in result.details


def test_bool_and_int_arrays():
    mask = np.array([True, False, True, True])
    flipped = mask.copy()
    flipped[2] = False
    result = _compare.compare_arrays(mask, flipped, real=3)
    assert not result.identical and result.details == (
        "bool (4,); real 3: 1 differ, first (2,); padding 1: 0 differ"
    )
    ints = np.array([1, 2, 3], dtype=np.int32)
    result = _compare.compare_arrays(ints, ints + np.array([0, 0, 5], dtype=np.int32))
    assert "max abs 5, max rel 0.625" in result.details


@pytest.mark.parametrize(
    "new, old, details",
    [
        (np.zeros(3, np.int32), np.zeros(3, np.int64), "int32 (3,) vs int64 (3,)"),
        (np.zeros(3), np.zeros(4), "float64 (3,) vs float64 (4,)"),
        (None, np.zeros(3), "present only on the reference"),
        (np.zeros(3), None, "present only on the new route"),
    ],
)
def test_type_shape_and_presence(new, old, details):
    result = _compare.compare_arrays(new, old)
    assert not result.identical and result.details == details


def test_absent_on_both_routes():
    assert _compare.compare_arrays(None, None).identical


def test_scalars():
    assert _compare.compare_scalars(0.1, 0.1).identical
    assert "0x1.999999999999ap-4" in _compare.compare_scalars(0.1, 0.1).details
    assert not _compare.compare_scalars(0.0, -0.0).identical
    assert not _compare.compare_scalars(1, True).identical  # a bool is not an int here
    assert not _compare.compare_scalars(1, 1.0).identical
    assert _compare.compare_scalars(True, True).identical
    assert not _compare.compare(3, 4).identical


def test_strings_numpy_scalars_and_none():
    assert _compare.compare("icon_grid", "icon_grid").identical
    assert _compare.compare("icon_grid", "icon_grid").details == "str 'icon_grid'"
    assert not _compare.compare("icon_grid", "other").identical
    assert not _compare.compare("1", 1).identical
    assert _compare.compare(np.int32(7), np.int32(7)).identical
    assert not _compare.compare(np.int32(7), np.int32(8)).identical
    assert not _compare.compare(np.int32(7), np.int64(7)).identical  # another dtype
    assert _compare.compare(None, None).identical
    assert not _compare.compare(_compare.perturb("icon_grid"), "icon_grid").identical
    assert not _compare.compare(_compare.perturb(np.int32(7)), np.int32(7)).identical


def test_host_array_of_a_field():
    array = np.arange(5.0)
    field = gtx.as_field([dims.CellDim], array)
    assert np.array_equal(_compare.host_array(field), array)
    assert _compare.host_array(None) is None


# ---- the self-test perturbation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        np.linspace(0.5, 1.5, 6).reshape(3, 2, order="F"),
        np.array([True, False, True]),
        np.arange(4, dtype=np.int32),
        np.arange(4, dtype=np.float32),
    ],
)
def test_perturb_changes_one_element_of_a_copy(value):
    before = value.copy()
    perturbed = _compare.perturb(value)
    assert np.array_equal(value, before) and not np.shares_memory(value, perturbed)
    result = _compare.compare_arrays(perturbed, value, real=1)
    assert not result.identical and (result.real_differ, result.padding_differ) == (1, 0)
    if value.dtype.kind == "f":
        assert "max ulp 1" in result.details


def test_perturb_scalars():
    assert _compare.perturb(True) is False
    assert _compare.perturb(3) == 4
    assert _compare.perturb(1.0) == math.nextafter(1.0, math.inf)
    assert _compare.perturb(math.nan) == 0.0
    # an absent array becomes a present one, so that the check reports it
    assert not _compare.compare(_compare.perturb(None), None).identical


def test_communicators():
    new = _compare.Communicator(-2080374780, (0, 1), "CONGRUENT")
    old = _compare.Communicator(1140850688, (0, 1))
    result = _compare.compare(new, old)
    assert result.identical
    assert result.details == "MPI_Comm_compare CONGRUENT; handle 0x84000004 vs 0x44000000; 2 ranks"
    assert _compare.compare(_compare.Communicator(5, (0, 1), "IDENT"), old).identical
    for relation in ("SIMILAR", "UNEQUAL", None):
        assert not _compare.compare(_compare.Communicator(5, (0, 1), relation), old).identical
    other = _compare.compare(_compare.Communicator(5, (1, 0), "CONGRUENT"), old)
    assert not other.identical and other.details.endswith("; members (1, 0) vs (0, 1)")
    assert not _compare.compare(new, 1140850688).identical
    perturbed = _compare.perturb(new)
    assert perturbed.members == (1, 1) and perturbed.handle == new.handle
    assert not _compare.compare(perturbed, old).identical
    assert _compare.host_array(new) is new


def test_form():
    array = np.zeros((8, 3), dtype=np.int32, order="F")
    assert _compare.form(array) == "numpy int32 (8, 3) strides (1, 8)"
    assert _compare.form(np.ascontiguousarray(array)) == "numpy int32 (8, 3) strides (3, 1)"
    assert _compare.form(np.zeros((8, 1), order="C")) == "numpy float64 (8, 1) strides (1, 0)"
    field = gtx.as_field([dims.CellDim, dims.C2EDim], array)
    assert _compare.form(field).startswith("Field[Cell, C2E] over numpy int32 (8, 3) strides")
    assert _compare.form(None) == "None"
