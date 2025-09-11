# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import math
import re

import numpy as np
import pytest

import gt4py.next as gtx

from icon4py.model.testing import (
    definitions,
    test_utils as testing_test_utils,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import (
    base,
    gridfile,
    horizontal as h_grid,
    icon,
)
from icon4py.model.testing import (
    grid_utils as gridtest_utils,
    definitions,
)
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
    return gridtest_utils.get_grid_manager_from_experiment(
        definitions.Experiments.MCH_CH_R04B09, keep_skip_values=True, backend=None
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
def test_halo(grid, dim, marker):
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


@pytest.mark.parametrize(
    "grid_descriptor",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
)
@pytest.mark.parametrize("offset", (utils.horizontal_offsets()), ids=lambda x: x.value)
def test_when_keep_skip_value_then_neighbor_table_matches_config(grid_descriptor, offset, backend):
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=True, backend=backend).grid
    connectivity = grid.get_connectivity(offset)

    assert (
        np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()
    ) == icon._has_skip_values(offset, grid.config.limited_area)
    if not icon._has_skip_values(offset, grid.config.limited_area):
        assert connectivity.skip_value is None
    else:
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX


@pytest.mark.parametrize(
    "grid_descriptor",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
)
@pytest.mark.parametrize("dim", (utils.local_dims()))
def test_when_replace_skip_values_then_only_pentagon_points_remain(
    grid_descriptor: definitions.GridDescription, dim: gtx.Dimension, backend
):
    if dim == dims.V2E2VDim:
        pytest.skip("V2E2VDim is not supported in the current grid configuration.")
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=False, backend=backend).grid
    connectivity = grid.get_connectivity(dim.value)
    if dim in icon.CONNECTIVITIES_ON_PENTAGONS and not grid.limited_area:
        assert np.any(
            connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX
        ).item(), f"Connectivity {dim.value} for {grid_descriptor.name} should have skip values."
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX
    else:
        assert not np.any(
            connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX
        ).item(), f"Connectivity {dim.value} for {grid_descriptor.name} contains skip values, but none are expected."
        assert connectivity.skip_value is None


def _sphere_area(radius: float) -> float:
    return 4.0 * math.pi * radius**2.0


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level,num_cells,mean_cell_area,expected_num_cells,expected_mean_cell_area",
    [
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            0,
            None,
            None,
            20,
            _sphere_area(constants.EARTH_RADIUS) / 20,
        ),
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            1,
            None,
            None,
            20 * 4,
            _sphere_area(constants.EARTH_RADIUS) / (20 * 4),
        ),
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            2,
            None,
            None,
            20 * 16,
            _sphere_area(constants.EARTH_RADIUS) / (20 * 16),
        ),
        (base.GeometryType.ICOSAHEDRON, 2, 4, None, None, 20480, 24907282236.708576),
        (base.GeometryType.ICOSAHEDRON, 4, 9, None, None, 83886080, 6080879.45232143),
        (base.GeometryType.ICOSAHEDRON, 2, 4, 42, 123.456, 42, 123.456),
        (base.GeometryType.ICOSAHEDRON, 4, 9, None, 123.456, 83886080, 123.456),
        (base.GeometryType.ICOSAHEDRON, 4, 9, 42, None, 42, 12145265243042.658),
        (base.GeometryType.TORUS, 2, 0, 42, 123.456, 42, 123.456),
        (base.GeometryType.TORUS, None, None, 42, None, 42, None),
    ],
)
def test_global_grid_params(
    geometry_type,
    grid_root,
    grid_level,
    num_cells,
    mean_cell_area,
    expected_num_cells,
    expected_mean_cell_area,
):
    params = icon.GlobalGridParams(
        grid_shape=icon.GridShape(
            geometry_type=geometry_type,
            subdivision=(
                icon.GridSubdivision(root=grid_root, level=grid_level)
                if grid_root is not None
                else None
            ),
        ),
        global_num_cells=num_cells,  # TODO(msimberg):
        mean_cell_area=mean_cell_area,
    )
    assert geometry_type == params.geometry_type
    if geometry_type == base.GeometryType.TORUS:
        assert None == params.grid_shape.subdivision
    else:
        assert (
            icon.GridSubdivision(root=grid_root, level=grid_level) == params.grid_shape.subdivision
        )
    assert expected_num_cells == params.num_cells
    assert expected_mean_cell_area == params.mean_cell_area


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level",
    [
        (base.GeometryType.ICOSAHEDRON, None, None),
        (base.GeometryType.ICOSAHEDRON, 0, 0),
        (None, None, None),
    ],
)
def test_grid_shape_fail(geometry_type, grid_root, grid_level):
    with pytest.raises(ValueError):
        _ = icon.GridShape(
            geometry_type=geometry_type,
            subdivision=(
                icon.GridSubdivision(root=grid_root, level=grid_level)
                if grid_root is not None
                else None
            ),
        )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_descriptor, geometry_type, subdivision, global_num_cells, num_cells, mean_cell_area",
    [
        (
            definitions.Grids.R02B04_GLOBAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=2, level=4),
            20480,
            20480,
            24907282236.708576,
        ),
        (
            definitions.Grids.R02B07_GLOBAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=2, level=7),
            1310720,
            1310720,
            389176284.94852674,
        ),
        (
            definitions.Grids.R19_B07_MCH_LOCAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=19, level=7),
            118292480,
            283876,
            4119096.374920686,
        ),
        (
            definitions.Grids.MCH_OPR_R04B07_DOMAIN01,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=7),
            5242880,
            10700,
            87967127.69851978,
        ),
        (
            definitions.Grids.MCH_OPR_R19B08_DOMAIN01,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=19, level=8),
            473169920,
            44528,
            1029968.5064089653,
        ),
        (
            definitions.Grids.MCH_CH_R04B09_DSL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=9),
            83886080,
            20896,
            6080879.45232143,
        ),
        (
            definitions.Grids.TORUS_100X116_1000M,
            base.GeometryType.TORUS,
            None,
            None,
            23200,
            433012.7018922193,
        ),
        (
            definitions.Grids.TORUS_50000x5000,
            base.GeometryType.TORUS,
            None,
            None,
            1056,
            248515.0952090332,
        ),
    ],
)
def test_global_grid_params_from_grid_manager(
    grid_descriptor,
    backend,
    geometry_type,
    subdivision,
    global_num_cells,
    num_cells,
    mean_cell_area,
):
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=True, backend=backend).grid
    params = grid.global_properties
    assert params is not None
    assert params.geometry_type == geometry_type
    assert params.subdivision == subdivision
    assert params.global_num_cells == global_num_cells
    assert params.num_cells == num_cells
    # Depending on which method is used to calculate the mean cell area, the
    # result may be slightly different. Allow a bit of tolerance.
    # TODO: How much to allow? Or just use exact values based on current
    # implementation and update if implementation changes?
    assert np.isclose(params.mean_cell_area, mean_cell_area, rtol=5e-2)
