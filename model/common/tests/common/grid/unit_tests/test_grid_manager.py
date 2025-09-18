# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import typing
from collections.abc import Iterator

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.common.grid.gridfile
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition, halo
from icon4py.model.common.grid import (
    grid_manager as gm,
    gridfile,
    horizontal as h_grid,
    refinement as refin,
    vertical as v_grid,
)
from icon4py.model.testing import datatest_utils as dt_utils, definitions, test_utils


if typing.TYPE_CHECKING:
    import netCDF4

    from icon4py.model.testing import serialbox

try:
    import netCDF4
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)


from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    processor_props,
    ranked_data_path,
)

from .. import utils


MCH_CH_RO4B09_GLOBAL_NUM_CELLS = 83886080


ZERO_BASE = icon4py.model.common.grid.gridfile.ToZeroBasedIndexTransformation()
vertical = v_grid.VerticalGridConfig(num_levels=80)


# TODO @magdalena add test cases for hexagon vertices v2e2v
# v2e2v: grid,???


# v2e: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_v2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    seralized_v2e = grid_savepoint.v2e()
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_connectivity("V2E").asnumpy()
    assert has_invalid_index(v2e_table)
    _reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_grid_manager_refin_ctrl(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    refin_ctrl = utils.run_grid_manager(
        experiment.grid, keep_skip_values=True, backend=backend
    ).grid.refinement_control
    refin_ctrl_serialized = grid_savepoint.refin_ctrl(dim)
    assert np.all(
        refin_ctrl_serialized.ndarray
        == refin.convert_to_unnested_refinement_values(refin_ctrl[dim].ndarray, dim)
    )


# v2c: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_v2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_v2c = grid_savepoint.v2c()
    v2c_table = grid.get_connectivity("V2C").asnumpy()
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    assert has_invalid_index(v2c_table)
    _reset_invalid_index(serialized_v2c)

    assert np.allclose(v2c_table, serialized_v2c)


def _reset_invalid_index(index_array: np.ndarray) -> None:
    """
    Revert changes from mo_model_domimp_patches.

    Helper function to revert mo_model_domimp_patches.f90: move_dummies_to_end_idxblk.
    see:
    # ! Checks for the pentagon case and moves dummy cells to end.
    #  ! The dummy entry is either set to 0 or duplicated from the last one
    #  SUBROUTINE move_dummies_to_end(array, array_size, max_connectivity, duplicate)

    After reading the grid file ICON moves all invalid indices (neighbors not existing in the
    grid file) to the end of the neighbor list and replaces them with the "last valid neighbor index"
    it is up to the user then to ensure that any coefficients in neighbor some multiplied with
    these values are zero in order to "remove" them again from the sum.

    For testing we resubstitute those to the GridFile.INVALID_INDEX
    Args:
        index_array: the array where values the invalid values have to be reset

    Returns: an array where the spurious "last valid index" are replaced by GridFile.INVALID_INDEX

    """
    for i in range(0, index_array.shape[0]):
        _, index = np.unique(index_array[i, :], return_index=True)
        index_array[i, max(index) + 1 :] = gridfile.GridFile.INVALID_INDEX


# e2v: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_e2v = grid_savepoint.e2v()
    e2v_table = grid.get_connectivity("E2V").asnumpy()
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(e2v_table)
    assert np.allclose(e2v_table, serialized_e2v)


def has_invalid_index(ar: np.ndarray) -> np.bool_:
    return np.any(np.isin(ar, gridfile.GridFile.INVALID_INDEX))


# e2c : exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_e2c = grid_savepoint.e2c()
    e2c_table = grid.get_connectivity("E2C").asnumpy()
    assert has_invalid_index(serialized_e2c) == grid.limited_area
    assert has_invalid_index(e2c_table) == grid.limited_area
    assert np.allclose(e2c_table, serialized_e2c)


# c2e: serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_c2e = grid_savepoint.c2e()
    c2e_table = grid.get_connectivity("C2E").asnumpy()
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(c2e_table)
    assert np.allclose(c2e_table, serialized_c2e)


# c2e2c: exists in  serial, simple_mesh, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    assert np.allclose(
        grid.get_connectivity("C2E2C").asnumpy(),
        grid_savepoint.c2e2c(),
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e2cO(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid()
    assert np.allclose(
        grid.get_connectivity("C2E2CO").asnumpy(),
        serialized_grid.get_connectivity("C2E2CO").asnumpy(),
    )


# e2c2e (e2c2eo) - diamond: exists in serial, simple_mesh
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid()
    serialized_e2c2e = serialized_grid.get_connectivity("E2C2E").asnumpy()
    serialized_e2c2eO = serialized_grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(serialized_e2c2e) == grid.limited_area

    e2c2e_table = grid.get_connectivity("E2C2E").asnumpy()
    e2c2eO_table = grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(e2c2e_table) == grid.limited_area
    # ICON calculates diamond edges only from rl_start = 2 (lateral_boundary(dims.EdgeDim) + 1 for
    # boundaries all values are INVALID even though the half diamond exists (see mo_model_domimp_setup.f90 ll 163ff.)
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    # e2c2e in ICON (quad_idx) has a different neighbor ordering than the e2c2e constructed in grid_manager.py
    assert_up_to_order(e2c2e_table, serialized_e2c2e, start_index)
    assert_up_to_order(e2c2eO_table, serialized_e2c2eO, start_index)


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_ref = grid_savepoint.e2c2v()
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(dims.EdgeDim) + 1) to end(dims.EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    table = grid.get_connectivity("E2C2V").asnumpy()
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    # e2c2e in ICON (quad_idx) has a different neighbor ordering than the e2c2e constructed in grid_manager.py
    assert_up_to_order(table, serialized_ref, start_index)
    assert np.allclose(table[:, :2], grid.get_connectivity("E2V").asnumpy())


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    c2v = grid.get_connectivity("C2V").asnumpy()
    assert np.allclose(c2v, grid_savepoint.c2v())


@pytest.mark.parametrize(
    "grid_descriptor", [definitions.Grids.R02B04_GLOBAL, definitions.Grids.MCH_CH_R04B09_DSL]
)
@pytest.mark.with_netcdf
def test_grid_manager_grid_size(
    backend: gtx_typing.Backend, grid_descriptor: definitions.GridDescription
) -> None:
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=True, backend=backend).grid
    assert grid_descriptor.sizes["cell"] == grid.size[dims.CellDim]
    assert grid_descriptor.sizes["edge"] == grid.size[dims.EdgeDim]
    assert grid_descriptor.sizes["vertex"] == grid.size[dims.VertexDim]


def assert_up_to_order(
    table: np.ndarray,
    reference_table: np.ndarray,
    start_index: gtx.int = 0,  # type: ignore[name-defined]
) -> None:
    assert table.shape == reference_table.shape, "arrays need to have the same shape"
    reduced_table = table[start_index:, :]
    reduced_reference = reference_table[start_index:, :]
    for n in range(reduced_table.shape[0]):
        assert np.all(
            np.isin(reduced_table[n, :], reduced_reference[n, :])
        ), f"values in row {n+start_index} are not equal: {reduced_table[n, :]} vs ref= {reduced_reference[n, :]}."


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort() -> None:
    fname = "./unknown_grid.nc"
    with pytest.raises(FileNotFoundError) as error:
        manager = gm.GridManager(
            fname,
            v_grid.VerticalGridConfig(num_levels=80),
            icon4py.model.common.grid.gridfile.NoTransformation(),
        )
        manager(backend=None, keep_skip_values=True)
        assert error.value == 1


@pytest.mark.parametrize("size", [100, 1500, 20000])
@pytest.mark.with_netcdf
def test_gt4py_transform_offset_by_1_where_valid(size: int) -> None:
    trafo = gridfile.ToZeroBasedIndexTransformation()
    rng = np.random.default_rng()
    input_field = rng.integers(-1, size, size)
    offset = trafo(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.parametrize(
    "grid_descriptor, global_num_cells",
    [
        (definitions.Grids.R02B04_GLOBAL, definitions.Grids.R02B04_GLOBAL.sizes["cell"]),
        (definitions.Grids.MCH_CH_R04B09_DSL, MCH_CH_RO4B09_GLOBAL_NUM_CELLS),
    ],
)
def test_grid_manager_grid_level_and_root(
    grid_descriptor: definitions.GridDescription, global_num_cells: int, backend: gtx_typing.Backend
) -> None:
    assert (
        global_num_cells
        == utils.run_grid_manager(
            grid_descriptor, keep_skip_values=True, backend=backend
        ).grid.global_properties.num_cells
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "experiment",
    [definitions.Experiments.JW],
)
def test_grid_manager_eval_c2e2c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid()
    assert np.allclose(
        grid.get_connectivity("C2E2C2E").asnumpy(),
        serialized_grid.get_connectivity("C2E2C2E").asnumpy(),
    )
    assert grid.get_connectivity("C2E2C2E").asnumpy().shape == (grid.num_cells, 9)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize("dim", utils.main_horizontal_dims())
def test_grid_manager_start_end_index(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    serialized_grid = grid_savepoint.construct_icon_grid()
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    for domain in utils.global_grid_domains(dim):
        if (
            dim == dims.EdgeDim
            and domain.zone == h_grid.Zone.END
            and experiment.grid == definitions.Grids.R02B04_GLOBAL
        ):
            pytest.xfail(
                "FIXME: start_index in serialized data changed to 0 with unknown consequences, see also icon-exclaim output"
            )
        assert grid.start_index(domain) == serialized_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == serialized_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"

    for domain in utils.valid_boundary_zones_for_dim(dim):
        if not grid.limited_area:
            assert grid.start_index(domain) == 0
            assert grid.end_index(domain) == 0
        assert grid.start_index(domain) == serialized_grid.start_index(
            domain
        ), f"start index wrong for domain {domain}"
        assert grid.end_index(domain) == serialized_grid.end_index(
            domain
        ), f"end index wrong for domain {domain}"


@pytest.mark.datatest
def test_read_geometry_fields(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    cell_area = manager.geometry[gridfile.GeometryName.CELL_AREA]
    tangent_orientation = manager.geometry[gridfile.GeometryName.TANGENT_ORIENTATION]

    assert test_utils.dallclose(cell_area.asnumpy(), grid_savepoint.cell_areas().asnumpy())
    assert test_utils.dallclose(
        tangent_orientation.asnumpy(), grid_savepoint.tangent_orientation().asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize("dim", (dims.CellDim, dims.EdgeDim, dims.VertexDim))
def test_coordinates(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    lat = manager.coordinates[dim]["lat"]
    lon = manager.coordinates[dim]["lon"]
    assert test_utils.dallclose(lat.asnumpy(), grid_savepoint.lat(dim).asnumpy())
    assert test_utils.dallclose(lon.asnumpy(), grid_savepoint.lon(dim).asnumpy())


@pytest.mark.datatest
def test_tangent_orientation(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.tangent_orientation()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.TANGENT_ORIENTATION].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
def test_edge_orientation_on_vertex(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.vertex_edge_orientation()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX].asnumpy(),
        expected.asnumpy(),
    )


@pytest.mark.datatest
def test_dual_area(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.vertex_dual_area()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.DUAL_AREA].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
def test_edge_cell_distance(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.edge_cell_length()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry

    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_CELL_DISTANCE].asnumpy(),
        expected.asnumpy(),
        equal_nan=True,
    )


@pytest.mark.datatest
def test_cell_normal_orientation(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.edge_orientation()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry
    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.CELL_NORMAL_ORIENTATION].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
def test_edge_vertex_distance(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.edge_vert_length()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    geometry_fields = manager.geometry

    assert test_utils.dallclose(
        geometry_fields[gridfile.GeometryName.EDGE_VERTEX_DISTANCE].asnumpy(),
        expected.asnumpy(),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "grid_descriptor, expected",
    [
        (definitions.Grids.MCH_CH_R04B09_DSL, True),
        (definitions.Grids.R02B04_GLOBAL, False),
    ],
)
def test_limited_area_on_grid(grid_descriptor: definitions.GridDescription, expected: bool) -> None:
    grid = utils.run_grid_manager(grid_descriptor, keep_skip_values=True, backend=None).grid
    assert expected == grid.limited_area


@pytest.mark.parametrize(
    "grid_file",
    [
        (definitions.Grids.MCH_CH_R04B09_DSL),
        (definitions.Grids.R02B04_GLOBAL),
    ],
)
@pytest.mark.parametrize("dim", utils.horizontal_dims())
def test_decomposition_info_single_node(
    dim: gtx.Dimension,
    grid_file: definitions.GridDescription,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.construct_decomposition_info()
    gm = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    result = gm.decomposition_info
    assert np.all(result.local_index(dim) == expected.local_index(dim))
    assert np.all(result.global_index(dim) == expected.global_index(dim))
    assert np.all(result.owner_mask(dim) == expected.owner_mask(dim))
    assert np.all(result.halo_levels(dim) == expected.halo_levels(dim))
