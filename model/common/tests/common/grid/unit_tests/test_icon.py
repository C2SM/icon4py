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
from icon4py.model.common.utils import data_allocation as data_alloc
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
    "grid_descriptor",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
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
    "grid_descriptor",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
)
@pytest.mark.parametrize("dim", (utils.local_dims()))
def test_when_replace_skip_values_then_only_pentagon_points_remain(
    grid_descriptor: definitions.GridDescription,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
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
    "geometry_type,grid_root,grid_level,global_num_cells,num_cells,mean_cell_area,expected_global_num_cells,expected_num_cells,expected_mean_cell_area",
    [
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            0,
            None,
            None,
            None,
            20,
            20,
            _sphere_area(constants.EARTH_RADIUS) / 20,
        ),
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            1,
            None,
            None,
            None,
            20 * 4,
            20 * 4,
            _sphere_area(constants.EARTH_RADIUS) / (20 * 4),
        ),
        (
            base.GeometryType.ICOSAHEDRON,
            1,
            2,
            None,
            None,
            None,
            20 * 16,
            20 * 16,
            _sphere_area(constants.EARTH_RADIUS) / (20 * 16),
        ),
        (base.GeometryType.ICOSAHEDRON, 2, 4, None, None, None, 20480, 20480, 24907282236.708576),
        (base.GeometryType.ICOSAHEDRON, 4, 9, 765, None, None, 765, 765, 666798876088.6165),
        (base.GeometryType.ICOSAHEDRON, 2, 4, None, 42, 123.456, 20480, 42, 123.456),
        (base.GeometryType.ICOSAHEDRON, 4, 9, None, None, 123.456, 83886080, 83886080, 123.456),
        (base.GeometryType.ICOSAHEDRON, 4, 9, None, 42, None, 83886080, 42, 6080879.45232143),
        (base.GeometryType.TORUS, 2, 0, None, 42, 123.456, None, 42, 123.456),
        (base.GeometryType.TORUS, None, None, None, 42, None, None, 42, None),
    ],
)
def test_global_grid_params(
    geometry_type: base.GeometryType,
    grid_root: int | None,
    grid_level: int | None,
    global_num_cells: int | None,
    num_cells: int | None,
    mean_cell_area: float | None,
    expected_global_num_cells: int | None,
    expected_num_cells: int | None,
    expected_mean_cell_area: float | None,
) -> None:
    if grid_root is None:
        assert grid_level is None
    params = icon.GlobalGridParams(
        grid_shape=icon.GridShape(
            geometry_type=geometry_type,
            subdivision=(
                icon.GridSubdivision(root=grid_root, level=grid_level)  # type: ignore[arg-type]
                if grid_root is not None
                else None
            ),
        ),
        domain_length=42.0,
        domain_height=100.5,
        global_num_cells=global_num_cells,
        num_cells=num_cells,
        mean_edge_length=13.0,
        mean_dual_edge_length=None,
        mean_cell_area=mean_cell_area,
        mean_dual_cell_area=None,
    )
    assert geometry_type == params.geometry_type
    if geometry_type == base.GeometryType.TORUS:
        assert params.grid_shape is not None
        assert params.grid_shape.subdivision is None
    else:
        assert (
            icon.GridSubdivision(root=grid_root, level=grid_level) == params.grid_shape.subdivision  # type: ignore[arg-type, union-attr]
        )
    if geometry_type == base.GeometryType.TORUS:
        assert params.radius is None
        assert params.domain_length == 42.0
        assert params.domain_height == 100.5
    else:
        assert pytest.approx(params.radius) == constants.EARTH_RADIUS
        assert params.domain_length is None
        assert params.domain_height is None
    assert params.global_num_cells == expected_global_num_cells
    assert params.num_cells == expected_num_cells
    assert pytest.approx(params.mean_edge_length) == 13.0
    assert params.mean_dual_edge_length is None
    assert pytest.approx(params.mean_cell_area) == expected_mean_cell_area
    assert params.mean_dual_cell_area is None
    if expected_mean_cell_area is not None:
        assert pytest.approx(params.characteristic_length) == math.sqrt(expected_mean_cell_area)


@pytest.mark.parametrize(
    "geometry_type",
    [base.GeometryType.ICOSAHEDRON, base.GeometryType.TORUS],
)
def test_global_grid_params_from_fields(
    geometry_type: base.GeometryType,
    backend: gtx_typing.Backend,
) -> None:
    xp = data_alloc.import_array_ns(backend)

    # Means provided directly (higher priority than calculating from fields)
    params = icon.GlobalGridParams.from_fields(
        grid_shape=icon.GridShape(
            geometry_type=geometry_type, subdivision=icon.GridSubdivision(root=2, level=2)
        ),
        mean_edge_length=13.0,
        mean_dual_edge_length=14.0,
        mean_cell_area=15.0,
        mean_dual_cell_area=16.0,
        edge_lengths=xp.asarray([1.0, 2.0]),
        dual_edge_lengths=xp.asarray([2.0, 3.0]),
        cell_areas=xp.asarray([3.0, 4.0]),
        dual_cell_areas=xp.asarray([4.0, 5.0]),
        array_ns=xp,
    )
    assert pytest.approx(params.mean_edge_length) == 13.0
    assert pytest.approx(params.mean_dual_edge_length) == 14.0
    assert pytest.approx(params.mean_cell_area) == 15.0
    assert pytest.approx(params.mean_dual_cell_area) == 16.0

    # Means computed from fields
    params = icon.GlobalGridParams.from_fields(
        grid_shape=icon.GridShape(
            geometry_type=geometry_type, subdivision=icon.GridSubdivision(root=2, level=2)
        ),
        edge_lengths=xp.asarray([1.0, 2.0]),
        dual_edge_lengths=xp.asarray([2.0, 3.0]),
        cell_areas=xp.asarray([3.0, 4.0]),
        dual_cell_areas=xp.asarray([4.0, 5.0]),
        array_ns=xp,
    )
    assert pytest.approx(params.mean_edge_length) == 1.5
    assert pytest.approx(params.mean_dual_edge_length) == 2.5
    assert pytest.approx(params.mean_cell_area) == 3.5
    assert pytest.approx(params.mean_dual_cell_area) == 4.5


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level",
    [
        (base.GeometryType.ICOSAHEDRON, None, None),
        (base.GeometryType.ICOSAHEDRON, 0, 0),
        (None, None, None),
    ],
)
def test_grid_shape_fail(geometry_type: base.GeometryType, grid_root: int, grid_level: int) -> None:
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
    "grid_descriptor, geometry_type, subdivision, radius, domain_length, domain_height, global_num_cells, num_cells, mean_edge_length, mean_dual_edge_length, mean_cell_area, mean_dual_cell_area, characteristic_length",
    [
        (
            definitions.Grids.R02B04_GLOBAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=2, level=4),
            constants.EARTH_RADIUS,
            None,
            None,
            20480,
            20480,
            240221.1036647776,
            138710.63736114913,
            24906292887.251026,
            49802858653.68937,
            157817.27689721118,
        ),
        (
            definitions.Grids.R02B07_GLOBAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=2, level=7),
            constants.EARTH_RADIUS,
            None,
            None,
            1310720,
            1310720,
            30050.07607616417,
            17349.90054929857,
            389176284.94852674,
            778350194.5608561,
            19727.55141796687,
        ),
        (
            definitions.Grids.R19_B07_MCH_LOCAL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=19, level=7),
            constants.EARTH_RADIUS,
            None,
            None,
            118292480,
            283876,
            3092.8086192896153,
            1782.4707626479924,
            4119096.374920686,
            8192823.87559748,
            2029.555708750239,
        ),
        (
            definitions.Grids.MCH_OPR_R04B07_DOMAIN01,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=7),
            constants.EARTH_RADIUS,
            None,
            None,
            5242880,
            10700,
            14295.416301386269,
            8173.498324820434,
            87967127.69851978,
            170825432.57740065,
            9379.079256436624,
        ),
        (
            definitions.Grids.MCH_OPR_R19B08_DOMAIN01,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=19, level=8),
            constants.EARTH_RADIUS,
            None,
            None,
            473169920,
            44528,
            1546.76182117618,
            889.1206039451661,
            1029968.5064089653,
            2032098.7893505183,
            1014.8736406119558,
        ),
        (
            definitions.Grids.MCH_CH_R04B09_DSL,
            base.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=9),
            constants.EARTH_RADIUS,
            None,
            None,
            83886080,
            20896,
            3803.019140934253,
            2180.911493355989,
            6256048.940145881,
            12259814.063180268,
            2501.209495453326,
        ),
        (
            definitions.Grids.TORUS_100X116_1000M,
            base.GeometryType.TORUS,
            None,
            None,
            100000.0,
            100458.94683899487,
            None,
            23200,
            1000.0,
            577.3502691896258,
            433012.7018922193,
            866025.4037844389,
            658.0370064762462,
        ),
        (
            definitions.Grids.TORUS_50000x5000,
            base.GeometryType.TORUS,
            None,
            None,
            50000.0,
            5248.638810814779,
            None,
            1056,
            757.5757575757576,
            437.3865675678984,
            248515.09520903317,
            497030.1904180664,
            498.51288369412595,
        ),
    ],
)
def test_global_grid_params_from_grid_manager(
    grid_descriptor: definitions.GridDescription,
    backend: gtx_typing.Backend,
    geometry_type: base.GeometryType,
    subdivision: icon.GridSubdivision,
    radius: float,
    domain_length: float,
    domain_height: float,
    global_num_cells: int,
    num_cells: int,
    mean_edge_length: float,
    mean_dual_edge_length: float,
    mean_cell_area: float,
    mean_dual_cell_area: float,
    characteristic_length: float,
) -> None:
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=True, backend=backend).grid
    params = grid.global_properties
    assert params is not None
    assert pytest.approx(params.geometry_type) == geometry_type
    assert pytest.approx(params.subdivision) == subdivision
    assert pytest.approx(params.radius) == radius
    assert pytest.approx(params.domain_length) == domain_length
    assert pytest.approx(params.domain_height) == domain_height
    assert pytest.approx(params.global_num_cells) == global_num_cells
    assert pytest.approx(params.num_cells) == num_cells
    assert pytest.approx(params.mean_edge_length) == mean_edge_length
    assert pytest.approx(params.mean_dual_edge_length) == mean_dual_edge_length
    assert pytest.approx(params.mean_cell_area) == mean_cell_area
    assert pytest.approx(params.mean_dual_cell_area) == mean_dual_cell_area
    assert pytest.approx(params.characteristic_length) == characteristic_length
