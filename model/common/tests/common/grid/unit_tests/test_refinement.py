# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

from icon4py.model.common.grid import refinement as refin, horizontal as h_grid
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils, definitions as test_defs

from .. import utils
from ..fixtures import *  # noqa: F401, F403


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
    "grid_file, expected",
    [(test_defs.Grids.R02B04_GLOBAL.name, False), (test_defs.Grids.MCH_CH_R04B09_DSL.name, True)],
)
def test_is_local_area_grid_for_grid_files(grid_file, expected, dim, backend):
    grid = grid_utils.get_grid_manager_from_identifier(grid_file, 1, True, backend).grid
    xp = data_alloc.array_ns(device_utils.is_cupy_device(backend))
    refinement_field = grid.refinement_control[dim]
    limited_area = refin.is_limited_area_grid(refinement_field.ndarray, array_ns=xp)
    assert isinstance(limited_area, bool)
    assert expected == limited_area


@pytest.fixture
def start_indices(grid_savepoint) -> dict:
    return {
        dims.CellDim: grid_savepoint.cells_start_index(),
        dims.EdgeDim: grid_savepoint.edge_start_index(),
        dims.VertexDim: grid_savepoint.vertex_start_index(),
    }


@pytest.mark.xfail
@pytest.mark.parametrize("dim", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
@pytest.mark.parametrize(
    "grid_file, experiment",
    [(test_defs.Grids.MCH_CH_R04B09_DSL.name, test_defs.Experiments.MCH_CH_R04B09.name)],
)
def test_compute_start_index(dim, grid_file, start_indices, experiment):
    reference_start = start_indices.get(dim)
    grid = grid_utils.get_grid_manager_from_identifier(
        grid_file, num_levels=1, keep_skip_values=True, backend=None
    ).grid
    refinement_control_field = grid.refinement_control[dim]
    start_index = refin.compute_start_index(dim, refinement_control_field.ndarray)
    assert start_index.ndim == 1
    assert start_index.shape[0] == h_grid.GRID_REFINEMENT_SIZE[dim]
    domain = h_grid.domain(dim)
    assert (
        start_index[domain(h_grid.Zone.LATERAL_BOUNDARY)]
        == reference_start[domain(h_grid.Zone.LATERAL_BOUNDARY)]
    )
    assert (
        start_index[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)]
        == reference_start[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)]
    )
    assert (
        start_index[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)]
        == reference_start[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)]
    )
    assert (
        start_index[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)]
        == reference_start[domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4)]
    )
    assert start_index[domain(h_grid.Zone.NUDGING)] == reference_start[domain(h_grid.Zone.NUDGING)]
    assert (
        start_index[domain(h_grid.Zone.INTERIOR)] == reference_start[domain(h_grid.Zone.INTERIOR)]
    )
