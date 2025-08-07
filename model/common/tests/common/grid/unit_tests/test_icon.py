# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import re

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    gridfile,
    horizontal as h_grid,
    icon,
)
from icon4py.model.testing import datatest_utils as dt_utils, grid_utils as gridtest_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)

from .. import utils


@functools.cache
def grid_from_limited_area_grid_file() -> icon.IconGrid:
    return gridtest_utils.get_grid_manager_for_experiment(
        dt_utils.REGIONAL_EXPERIMENT, keep_skip_values=True, backend=None
    ).grid


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


@pytest.fixture(params=["serialbox", "file"])
def grid(icon_grid, request):
    if request.param == "serialbox":
        return icon_grid
    else:
        return grid_from_limited_area_grid_file()


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("marker", [h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2])
def test_halo(icon_grid, dim, marker):
    # For single node this returns an empty region - start and end index are the same see  also ./mpi_tests/test_icon.py
    domain = h_grid.domain(dim)(marker)
    assert grid.start_index(domain) == HALO_IDX[dim][0]
    assert grid.end_index(domain) == HALO_IDX[dim][1]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_local(dim, grid):
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    assert grid.start_index(domain) == 0
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("marker", lateral_boundary())
def test_lateral_boundary(grid, dim, marker):
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
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_end(grid, dim):
    domain = h_grid.domain(dim)(h_grid.Zone.END)
    assert grid.start_index(domain) == grid.size[dim]
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("marker", nudging())
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_nudging(grid, dim, marker):
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    if dim == dims.VertexDim or (dim == dims.CellDim and num > 1):
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
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_interior(grid, dim):
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


@pytest.mark.parametrize("dim", ("C2CE", "E2EC", "E2ECV", "C2CEC", "C2CE", "C2CECEC"))
@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
def test_1d_sparse_fields_connectivities_have_no_skip_values(grid_file, dim, backend):
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    connectivity = grid.get_connectivity(dim)
    assert (
        connectivity.skip_value is None
    ), f"Connectivity {dim} for {grid_file} should not have skip values."
    assert (
        connectivity.asnumpy().min() >= 0
    ), f"Connectivity {dim} for {grid_file} should not have negative values."


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
@pytest.mark.parametrize("offset", (utils.horizontal_offsets()), ids=lambda x: x.value)
def test_when_keep_skip_value_then_neighbor_table_matches_config(
    grid_file, offset, backend, caplog
):
    caplog.set_level(logging.DEBUG)
    if offset.source in utils.one_dimensional_sparse_dims():
        pytest.skip("Sparse 1d connectivities never have skip_values in our setup.")
    grid = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend).grid
    connectivity = grid.get_connectivity(offset)

    assert (
        np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()
    ) == icon._has_skip_values(offset, grid.config.limited_area)
    if not icon._has_skip_values(offset, grid.config.limited_area):
        assert connectivity.skip_value is None
    else:
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX


@pytest.mark.parametrize("grid_file", (dt_utils.REGIONAL_EXPERIMENT, dt_utils.R02B04_GLOBAL))
@pytest.mark.parametrize("dim", (utils.local_dims()))
def test_when_replace_skip_values_then_only_pentagon_points_remain(grid_file, dim, backend, caplog):
    caplog.set_level(logging.DEBUG)
    if dim == dims.V2E2VDim:
        pytest.skip("V2E2VDim is not supported in the current grid configuration.")
    grid = utils.run_grid_manager(grid_file, keep_skip_values=False, backend=backend).grid
    connectivity = grid.get_connectivity(dim.value)
    if dim in icon.CONNECTIVITIES_ON_PENTAGONS and not grid.limited_area:
        assert np.any(
            connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX
        ).item(), f"Connectivity {dim.value} for {grid_file} should have skip values."
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX
    else:
        assert (
            not np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()
        ), f"Connectivity {dim.value} for {grid_file} contains skip values, but none are expected."
        assert connectivity.skip_value is None


@pytest.mark.parametrize(
    "grid_root,grid_level,num_cells,mean_cell_area,expected_num_cells,expected_mean_cell_area",
    [
        (2, 4, None, None, 20480, 24907282236.708576),
        (4, 9, None, None, 83886080, 6080879.45232143),
        (2, 4, 42, 123.456, 42, 123.456),
        (4, 9, None, 123.456, 83886080, 123.456),
        (4, 9, 42, None, 42, 12145265243042.658),
    ],
)
def test_global_grid_params(
    grid_root, grid_level, num_cells, mean_cell_area, expected_num_cells, expected_mean_cell_area
):
    params = icon.GlobalGridParams(grid_root, grid_level, num_cells, mean_cell_area)
    assert expected_num_cells == params.num_cells
    assert expected_mean_cell_area == params.mean_cell_area


@pytest.mark.parametrize(
    "grid_root,grid_level,num_cells,mean_cell_area,expected_num_cells,expected_mean_cell_area",
    [
        (2, 4, 42, 123.456, 42, 123.456),
        (4, 9, None, 123.456, 83886080, 123.456),
        (0, 0, 42, 123.456, 42, 123.456),
        (None, None, 42, 123.456, 42, 123.456),
    ],
)
def test_global_grid_params_from_mean_cell_area(
    grid_root, grid_level, num_cells, mean_cell_area, expected_num_cells, expected_mean_cell_area
):
    params = icon.GlobalGridParams.from_mean_cell_area(
        mean_cell_area, num_cells=num_cells, root=grid_root, level=grid_level
    )
    assert expected_num_cells == params.num_cells
    assert expected_mean_cell_area == params.mean_cell_area
