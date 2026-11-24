# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import field, quantity, state, type_alias as ta, utils as common_utils
from icon4py.model.common.grid import base as base_grid

from tests.common.fixtures import grid
from tests.common.unit_tests.utils import Pressure, Salt, TendencyOfPressure, Vt


class Column(state.State):
    pressure: field.Field[Pressure]
    salt: field.Field[Salt]
    dtime: float


class Tracers(state.State):
    pressure: field.Field[Pressure]
    salt: field.Field[Salt] | None = None
    vt: field.Field[Vt] | None = None


class Budget(state.State):
    pressure: field.Field[Pressure]
    tend_pressure: field.Field[TendencyOfPressure]
    salt: field.Field[Salt] | None = None


def test_allocate_takes_the_plain_leaves_by_keyword(grid: base_grid.Grid) -> None:
    column = state.allocate(Column, grid, None, dtime=60.0)
    assert column.dtime == 60.0
    assert list(state.fields(column)) == ["pressure", "salt"]
    # a plain leaf without a default must be given
    with pytest.raises(TypeError, match="dtime"):
        state.allocate(Column, grid, None)


def test_fields_of_a_kind_and_without_one(grid: base_grid.Grid) -> None:
    budget = state.allocate(Budget, grid, None)
    assert list(state.fields(budget, kind=quantity.Tendency)) == ["tend_pressure"]
    assert list(state.fields(budget, kind=Pressure)) == ["pressure"]
    assert list(state.fields(budget, exclude=quantity.Tendency)) == ["pressure", "salt"]


def test_copy_writes_the_destination_in_place(grid: base_grid.Grid) -> None:
    src = state.allocate(Tracers, grid, None, only=("vt",), fill=lambda name, _: 2.0)
    dst = state.allocate(Tracers, grid, None, only=("vt",))

    assert state.copy(src, dst) is dst

    assert dst.salt is None and dst.vt is not src.vt
    np.testing.assert_array_equal(dst.vt.asnumpy(), 2.0)  # type: ignore[union-attr]
    # a leaf present on one side only is a mismatch
    with pytest.raises(ValueError, match="present leaves must match"):
        state.copy(src, state.allocate(Tracers, grid, None))


def test_copy_of_a_kind_writes_and_compares_only_those_leaves(grid: base_grid.Grid) -> None:
    src = state.allocate(Budget, grid, None, fill=lambda name, _: 1.0)
    # `salt` is present on one side only, and not a tendency: it is not compared
    dst = state.allocate(Budget, grid, None, only=("pressure", "tend_pressure"))

    state.copy(src, dst, kind=quantity.Tendency)

    np.testing.assert_array_equal(dst.tend_pressure.asnumpy(), 1.0)
    np.testing.assert_array_equal(dst.pressure.asnumpy(), 0.0)


def test_optional_leaves_left_none_are_not_fields(grid: base_grid.Grid) -> None:
    tracers = state.allocate(Tracers, grid, allocator=None, only=("vt",))
    assert tracers.salt is None
    assert list(state.fields(tracers)) == ["pressure", "vt"]
    assert state.fields(tracers)["vt"] is tracers.vt
    assert list(state.fields(state.allocate(Tracers, grid, allocator=None))) == [
        "pressure",
        "salt",
        "vt",
    ]


def test_allocate_follows_the_declared_dims(grid: base_grid.Grid) -> None:
    class Fields(state.State):
        pressure: field.Field[Pressure]
        salt: field.Field[Salt]
        salt_pc: common_utils.PredictorCorrectorPair[field.Field[Salt]]

    fields = state.allocate(Fields, grid, allocator=None, fill=lambda name, shape: float(len(name)))
    assert fields.pressure.ndarray.shape == (grid.num_cells, grid.num_levels)
    assert fields.salt.ndarray.shape == (grid.num_cells,)
    assert np.all(np.asarray(fields.salt.ndarray) == 4.0)
    # a pair gets one field per member, which `fields` skips
    assert fields.salt_pc.predictor is not fields.salt_pc.corrector
    assert np.all(np.asarray(fields.salt_pc.corrector.ndarray) == 7.0)
    assert list(state.fields(fields)) == ["pressure", "salt"]
