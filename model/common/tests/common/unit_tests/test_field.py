# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import pickle

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import (
    dimension as dims,
    field,
    field_type_aliases as fa,
    framework as fw,
    type_alias as ta,
)
from icon4py.model.common.grid import base as base_grid

from tests.common.fixtures import grid
from tests.common.unit_tests.utils import CELL_K, Pressure


def test_field_is_the_gt4py_field_it_tags(grid: base_grid.Grid) -> None:
    data = gtx.zeros({dims.CellDim: grid.num_cells, dims.KDim: grid.num_levels}, dtype=ta.wpfloat)
    pressure = field.Field(Pressure, data)
    assert isinstance(pressure, gtx.Field)
    assert pressure.quantity is Pressure and pressure.domain.dims == CELL_K
    assert pressure.ndarray is data.ndarray and pressure.domain == data.domain
    retagged = field.Field(Pressure, pressure)
    assert retagged.quantity is Pressure and retagged.ndarray is data.ndarray


def test_fields_derived_from_a_field_are_untagged(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    assert not isinstance(pressure + 1.0, field.Field)
    assert not isinstance(pressure[dims.CellDim(0) : dims.CellDim(2)], field.Field)


@gtx.field_operator
def _double(field: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return 2.0 * field


def test_a_program_takes_a_field_as_is(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    doubled = fw.zeros(Pressure, grid, allocator=None)
    np.asarray(pressure.ndarray)[...] = 3.0
    _double(pressure, out=doubled, offset_provider={})
    assert np.all(np.asarray(doubled.ndarray) == 6.0)


def test_field_copies_keep_the_quantity(grid: base_grid.Grid) -> None:
    pressure = fw.zeros(Pressure, grid, allocator=None)
    for copied in (
        copy.copy(pressure),
        copy.deepcopy(pressure),
        pickle.loads(pickle.dumps(pressure)),
    ):
        assert isinstance(copied, field.Field) and copied.quantity is Pressure
