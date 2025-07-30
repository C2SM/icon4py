# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

import icon4py.model.common.grid.refinement as refin
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils
from icon4py.model.testing.fixtures import backend

from .. import utils


def out_of_range(dim: gtx.Dimension):
    lower = range(-36, refin._UNORDERED[dim][1])
    for v in lower:
        yield v

    upper = range(refin._MAX_ORDERED[dim] + 1, 36)
    for v in upper:
        yield v


def refinement_value(dim: gtx.Dimension):
    lower = refin._UNORDERED[dim][1]
    upper = refin._MAX_ORDERED[dim]
    for v in range(lower, upper):
        yield v


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_ordered(dim):
    for value in refinement_value(dim):
        ordered = refin.RefinementValue(dim, value)

        if ordered.value in refin._UNORDERED[dim]:
            assert not ordered.is_ordered()
        else:
            assert ordered.is_ordered()


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_nested(dim):
    for value in refinement_value(dim):
        nested = refin.RefinementValue(dim, value)
        if nested.value < 0:
            assert nested.is_nested()
        else:
            assert not nested.is_nested()


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_valid_refinement_values(dim):
    for value in out_of_range(dim):
        with pytest.raises(AssertionError):
            refin.RefinementValue(dim, value)


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize(
    "grid_file, expected", [(dt_utils.R02B04_GLOBAL, False), (dt_utils.REGIONAL_EXPERIMENT, True)]
)
def test_is_local_area_grid_for_grid_files(grid_file, expected, dim, backend):
    grid = grid_utils.get_grid_manager(grid_file, 1, True, backend).grid
    xp = data_alloc.array_ns(device_utils.is_cupy_device(backend))
    refinement_field = grid.refinement_control[dim]
    limited_area = refin.is_limited_area_grid(refinement_field.ndarray, array_ns=xp)
    assert isinstance(limited_area, bool)
    assert expected == limited_area
