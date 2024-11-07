# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import re

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    grid_manager as gm,
    horizontal as h_grid,
    icon,
    vertical as v_grid,
)

from . import utils


@functools.cache
def grid_from_file() -> icon.IconGrid:
    file_name = utils.resolve_file_from_gridfile_name("mch_ch_r04b09_dsl")
    manager = gm.GridManager(
        gm.ToZeroBasedIndexTransformation(), str(file_name), v_grid.VerticalGridConfig(1)
    )
    manager()
    return manager.grid


def lateral_boundary():
    for marker in h_grid.Zone.__members__.values():
        if "lb" in marker.value:
            yield marker


def nudging():
    for marker in h_grid.Zone.__members__.values():
        if "nudging" in marker.value:
            yield marker


LATERAL_BOUNDARY_IDX = {
    dims.CellDim: [0, 850, 1688, 2511, 3316, 4104],
    dims.EdgeDim: [0, 428, 1278, 1700, 2538, 2954, 3777, 4184, 4989, 5387, 6176],
    dims.VertexDim: [0, 428, 850, 1266, 1673, 2071],
}

NUDGING_IDX = {
    dims.CellDim: [3316, 4104],
    dims.EdgeDim: [4989, 5387, 6176],
}
HALO_IDX = {
    dims.CellDim: [20896, 20896],
    dims.EdgeDim: [31558, 31558],
    dims.VertexDim: [10663, 10663],
}
INTERIOR_IDX = {
    dims.CellDim: [4104, HALO_IDX[dims.CellDim][0]],
    dims.EdgeDim: [6176, HALO_IDX[dims.EdgeDim][0]],
    dims.VertexDim: [2071, HALO_IDX[dims.VertexDim][0]],
}


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("dim", utils.horizontal_dim())
@pytest.mark.parametrize("marker", [h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2])
def test_halo(icon_grid, source, dim, marker):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    # For single node this returns an empty region - start and end index are the same see  also ./mpi_tests/test_icon.py
    domain = h_grid.domain(dim)(marker)
    assert grid.start_index(domain) == HALO_IDX[dim][0]
    assert grid.end_index(domain) == HALO_IDX[dim][1]


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_local(dim, source, icon_grid):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    assert grid.start_index(domain) == 0
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("dim", utils.horizontal_dim())
@pytest.mark.parametrize("marker", lateral_boundary())
def test_lateral_boundary(icon_grid, source, dim, marker):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    if num > 4 and dim in (dims.VertexDim, dims.CellDim):
        with pytest.raises(AssertionError) as e:
            h_grid.domain(dim)(marker)
            e.match(f"Invalid marker '{marker}' for dimension")
    else:
        domain = h_grid.domain(dim)(marker)
        start_index = grid.start_index(domain)
        end_index = grid.end_index(domain)
        assert start_index == LATERAL_BOUNDARY_IDX[dim][num - 1]
        assert end_index == LATERAL_BOUNDARY_IDX[dim][num]


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_end(icon_grid, source, dim):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    domain = h_grid.domain(dim)(h_grid.Zone.END)
    assert grid.start_index(domain) == grid.size[dim]
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("marker", nudging())
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_nudging(icon_grid, source, dim, marker):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    if dim == dims.VertexDim or dim == dims.CellDim and num > 1:
        with pytest.raises(AssertionError) as e:
            h_grid.domain(dim)(marker)
            e.match(f"Invalid marker '{marker}' for dimension")
    else:
        domain = h_grid.domain(dim)(marker)
        start_index = grid.start_index(domain)
        end_index = grid.end_index(domain)
        assert start_index == NUDGING_IDX[dim][num - 1]
        assert end_index == NUDGING_IDX[dim][num]


@pytest.mark.datatest
@pytest.mark.parametrize("source", ("serialbox", "file"))
@pytest.mark.parametrize("dim", utils.horizontal_dim())
def test_interior(icon_grid, source, dim):
    # working around the fact that fixtures cannot be used in parametrized functions
    grid = icon_grid if source == "serialbox" else grid_from_file()
    domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
    start_index = grid.start_index(domain)
    end_index = grid.end_index(domain)
    assert start_index == INTERIOR_IDX[dim][0]
    assert end_index == INTERIOR_IDX[dim][1]


@pytest.mark.datatest
def test_grid_size(icon_grid):
    assert 10663 == icon_grid.size[dims.VertexDim]
    assert 20896 == icon_grid.size[dims.CellDim]
    assert 31558 == icon_grid.size[dims.EdgeDim]
