# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest
import numpy as np
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import refinement as refin,horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils, definitions as test_defs
from icon4py.model.testing.fixtures import backend

from .. import utils
from ..utils import main_horizontal_dims


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
    "grid_file, expected", [(test_defs.Grids.R02B04_GLOBAL.name, False), (test_defs.Grids.MCH_OPR_R04B07_DOMAIN01.name, True)]
)
def test_is_local_area_grid_for_grid_files(grid_file, expected, dim, backend):
    grid = grid_utils.get_grid_manager_from_identifier(grid_file, 1, True, backend).grid
    xp = data_alloc.array_ns(device_utils.is_cupy_device(backend))
    refinement_field = grid.refinement_control[dim]
    limited_area = refin.is_limited_area_grid(refinement_field.ndarray, array_ns=xp)
    assert isinstance(limited_area, bool)
    assert expected == limited_area



@pytest.mark.parametrize("dim, zone, expected", [(
                                            dims.CellDim, h_grid.Zone.LATERAL_BOUNDARY, 0),
                                            (dims.CellDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, 629),
                                            (dims.CellDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3, 1244),
                                            (dims.CellDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4, 1843),
                                            (dims.CellDim, h_grid.Zone.NUDGING, 2424),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY, 0),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, 318),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3, 947),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4, 1258),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5, 1873),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6, 2177),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7, 2776),
                                            (dims.EdgeDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8, 3071),
                                            (dims.EdgeDim, h_grid.Zone.NUDGING, 3652),
                                            (dims.EdgeDim, h_grid.Zone.NUDGING_LEVEL_2, 3938),
                                            (dims.VertexDim, h_grid.Zone.LATERAL_BOUNDARY, 0),
                                            (dims.VertexDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, 318),
                                            (dims.VertexDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3, 629),
                                            (dims.VertexDim, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,933),
])
def test_compute_start_index_limited_area_grid(dim, zone, expected):
    grid = grid_utils.get_grid_manager_from_identifier(test_defs.Grids.MCH_OPR_R04B07_DOMAIN01.name, 1, True, None).grid
    refinement_field = grid.refinement_control
    domain = h_grid.domain(dim)(zone)
    start_index = refin.compute_start_index(domain, refinement_field, array_ns=np)
    assert isinstance(start_index, gtx.int32)
    assert start_index == expected, f"Expected start index {expected} for {dim} in {zone}, but got {start_index}"


@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("zone", (h_grid.Zone.LATERAL_BOUNDARY, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7, h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8))
def test_compute_start_index_global_grid(dim, zone):
    grid = grid_utils.get_grid_manager_from_identifier(test_defs.Grids.R02B04_GLOBAL.name, 1, True, None).grid
    refinement_field = grid.refinement_control
    # TODO(halungge): for debugging - remove this
    reference_start_indices = grid._start_indices
    reference_end_indices = grid._end_indices
    domain = h_grid.domain(dim)(h_grid.Zone.LATERAL_BOUNDARY)
    start_index = refin.compute_start_index(domain, refinement_field, array_ns=np)
    assert isinstance(start_index, gtx.int32)
    assert start_index == grid.size[dim]
    # grid file start_idx_[c, v, e] is set to grid.size[dim] for global grids, but current grid._start_indices[domain], grid._end_indices[domain] is set to
    # TODO(halungge): for debugging - remove this
    # assert start_index == reference_start_indices[domain]
