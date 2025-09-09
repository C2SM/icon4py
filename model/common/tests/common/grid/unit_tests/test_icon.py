# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import functools
import math
import re
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, gridfile, horizontal as h_grid, icon
from icon4py.model.testing import definitions, grid_utils as gridtest_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    processor_props,
    ranked_data_path,
)

from .. import utils


if TYPE_CHECKING:
    from collections.abc import Iterator

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@pytest.fixture(scope="module")
def experiment() -> definitions.Experiment:
    """The module uses hard-coded references for the MCH_CH_R04B09 experiment."""
    return definitions.Experiments.MCH_CH_R04B09


@functools.cache
def grid_from_limited_area_grid_file() -> icon.IconGrid:
    return gridtest_utils.get_grid_manager_from_experiment(
        definitions.Experiments.MCH_CH_R04B09, keep_skip_values=True, backend=None
    ).grid


def lateral_boundary() -> Iterator[h_grid.Zone]:
    for marker in h_grid.Zone.__members__.values():
        if "lb" in marker.value:
            yield marker


def nudging() -> Iterator[h_grid.Zone]:
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
def grid(icon_grid: base_grid.Grid, request: pytest.FixtureRequest) -> base_grid.Grid:
    if request.param == "serialbox":
        return icon_grid
    else:
        return grid_from_limited_area_grid_file()


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("marker", [h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2])
def test_halo(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
    # For single node this returns an empty region - start and end index are the same see  also ./mpi_tests/test_icon.py
    domain = h_grid.domain(dim)(marker)
    assert grid.start_index(domain) == HALO_IDX[dim][0]
    assert grid.end_index(domain) == HALO_IDX[dim][1]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_local(dim: gtx.Dimension, grid: base_grid.Grid) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    assert grid.start_index(domain) == 0
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
@pytest.mark.parametrize("marker", lateral_boundary())
def test_lateral_boundary(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
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
def test_end(grid: base_grid.Grid, dim: gtx.Dimension) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.END)
    assert grid.start_index(domain) == grid.size[dim]
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("marker", nudging())
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_nudging(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
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
def test_interior(grid: base_grid.Grid, dim: gtx.Dimension) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
    start_index = grid.start_index(domain)
    end_index = grid.end_index(domain)
    assert start_index == INTERIOR_IDX[dim][0]
    assert end_index == INTERIOR_IDX[dim][1]


@pytest.mark.datatest
def test_grid_size(icon_grid: base_grid.Grid) -> None:
    assert icon_grid.size[dims.VertexDim] == 10663
    assert icon_grid.size[dims.CellDim] == 20896
    assert icon_grid.size[dims.EdgeDim] == 31558


@pytest.mark.parametrize(
    "grid_descriptor", (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL)
)
@pytest.mark.parametrize("offset", (utils.horizontal_offsets()), ids=lambda x: x.value)
def test_when_keep_skip_value_then_neighbor_table_matches_config(
    grid_descriptor: definitions.GridDescription,
    offset: gtx.FieldOffset,
    backend: gtx_typing.Backend,
) -> None:
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
    "grid_descriptor", (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL)
)
@pytest.mark.parametrize("dim", (utils.local_dims()))
def test_when_replace_skip_values_then_only_pentagon_points_remain(
    grid_descriptor: definitions.GridDescription, dim: gtx.Dimension, backend: gtx_typing.Backend
) -> None:
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
        (base.GeometryType.TORUS, 2, 0, 42, None, 42, None),
        (base.GeometryType.TORUS, None, None, 42, None, 42, None),
    ],
)
def test_global_grid_params(
    geometry_type: base.GeometryType,
    grid_root: int | None,
    grid_level: int | None,
    num_cells: int | None,
    mean_cell_area: float | None,
    expected_num_cells: int | None,
    expected_mean_cell_area: float | None,
) -> None:
    if grid_root is None:
        assert grid_level is None
    params = icon.GlobalGridParams(
        icon.GridShape(
            geometry_type=geometry_type,
            subdivision=icon.GridSubdivision(root=grid_root, level=grid_level)  # type: ignore[arg-type]
            if grid_root is not None
            else None,
        ),
        num_cells,
        mean_cell_area,
    )
    assert geometry_type == params.geometry_type
    if geometry_type == base.GeometryType.TORUS:
        assert params.grid_shape is not None
        assert params.grid_shape.subdivision is None
    else:
        assert (
            icon.GridSubdivision(root=grid_root, level=grid_level) == params.grid_shape.subdivision  # type: ignore[arg-type, union-attr]
        )
    assert expected_num_cells == params.num_cells
    if geometry_type == base.GeometryType.TORUS:
        with pytest.raises(NotImplementedError) as e:
            assert expected_mean_cell_area == params.mean_cell_area
            e.match("mean_cell_area is not implemented for GeometryType.TORUS")
    else:
        assert expected_mean_cell_area == params.mean_cell_area


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level,num_cells,mean_cell_area",
    [
        (base.GeometryType.ICOSAHEDRON, 0, 0, 42, 123.456),
    ],
)
def test_global_grid_params_fail(
    geometry_type: base.GeometryType,
    grid_root: int,
    grid_level: int,
    num_cells: int,
    mean_cell_area: float,
) -> None:
    with pytest.raises(ValueError):
        _ = icon.GlobalGridParams(
            icon.GridShape(
                geometry_type=geometry_type,
                subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),
            ),
            num_cells,
            mean_cell_area,
        )


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level,num_cells,mean_cell_area,expected_num_cells,expected_mean_cell_area",
    [
        (base.GeometryType.ICOSAHEDRON, 2, 4, 42, 123.456, 42, 123.456),
        (base.GeometryType.ICOSAHEDRON, 4, 9, None, 123.456, 83886080, 123.456),
        (base.GeometryType.TORUS, 2, 0, 42, 123.456, 42, 123.456),
    ],
)
def test_global_grid_params_from_mean_cell_area(
    geometry_type: base.GeometryType,
    grid_root: int,
    grid_level: int,
    num_cells: int | None,
    mean_cell_area: float,
    expected_num_cells: int,
    expected_mean_cell_area: float,
) -> None:
    params = icon.GlobalGridParams.from_mean_cell_area(
        mean_cell_area,
        num_cells=num_cells,
        grid_shape=icon.GridShape(
            geometry_type=geometry_type,
            subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),
        )
        if grid_root is not None and grid_level is not None
        else None,
    )
    assert expected_num_cells == params.num_cells
    assert expected_mean_cell_area == params.mean_cell_area
