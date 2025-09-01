# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import (
    base,
    geometry_attributes as geometry_attrs,
    horizontal as h_grid,
)
from icon4py.model.common.grid.gridfile import GridFile
from icon4py.model.common.interpolation import rbf_interpolation as rbf
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    grid_utils as gridtest_utils,
    test_utils as test_helpers,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_cells(
    grid_file, experiment, grid_savepoint, icon_grid, backend
):
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(grid_file, 1, True, backend)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_cells(grid)
    assert offset_table.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.c2e2c2e()
    assert offset_table.shape == offset_table_savepoint.shape

    # Savepoint neighbors before start index may not be populated correctly,
    # ignore them.
    start_index = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_edges(
    grid_file, experiment, grid_savepoint, icon_grid, backend
):
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(grid_file, 1, True, backend)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_edges(grid)
    assert offset_table.shape == (
        grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.e2c2e()
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT),
    ],
)
def test_construct_rbf_matrix_offsets_tables_for_vertices(
    experiment, grid_file, grid_savepoint, icon_grid, backend
):
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(grid_file, 1, True, backend)
    grid = grid_manager.grid
    offset_table = rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid)
    assert offset_table.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = grid_savepoint.v2e()
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = icon_grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Make sure invalid neighbors are represented the same way.
        _, index = np.unique(offset_table_savepoint[i, :], return_index=True)
        offset_table_savepoint[i, max(index) + 1 :] = GridFile.INVALID_INDEX

        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-9),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 3e-2),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, dt_utils.WEISMAN_KLEMP_EXPERIMENT, 3e-2),
    ],
)
def test_rbf_interpolation_coeffs_cell(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment, atol
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.CELL

    horizontal_start = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    assert horizontal_start < grid.num_cells

    rbf_vec_coeff_c1, rbf_vec_coeff_c2 = rbf.compute_rbf_interpolation_coeffs_cell(
        geometry.get(geometry_attrs.CELL_LAT).ndarray,
        geometry.get(geometry_attrs.CELL_LON).ndarray,
        geometry.get(geometry_attrs.CELL_CENTER_X).ndarray,
        geometry.get(geometry_attrs.CELL_CENTER_Y).ndarray,
        geometry.get(geometry_attrs.CELL_CENTER_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        rbf.construct_rbf_matrix_offsets_tables_for_cells(grid),
        rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        grid.global_properties.geometry_type,
        # TODO(msimberg): Write this to config
        rbf.compute_default_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim)
        if grid.global_properties.geometry_type is base.GeometryType.ICOSAHEDRON
        else grid.mean_dual_edge_length,
        horizontal_start,
        grid.global_properties.domain_length,
        grid.global_properties.domain_height,
        array_ns=data_alloc.import_array_ns(backend),
    )

    rbf_vec_coeff_c1_ref = interpolation_savepoint.rbf_vec_coeff_c1().ndarray
    rbf_vec_coeff_c2_ref = interpolation_savepoint.rbf_vec_coeff_c2().ndarray

    assert rbf_vec_coeff_c1.shape == rbf_vec_coeff_c1_ref.shape
    assert rbf_vec_coeff_c2.shape == rbf_vec_coeff_c2_ref.shape
    assert rbf_vec_coeff_c1_ref.shape == (
        icon_grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert rbf_vec_coeff_c2_ref.shape == (
        icon_grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c1[horizontal_start:],
        rbf_vec_coeff_c1_ref[horizontal_start:],
        atol=atol,
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c2[horizontal_start:],
        rbf_vec_coeff_c2_ref[horizontal_start:],
        atol=atol,
    )


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 3e-10),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 3e-3),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, dt_utils.WEISMAN_KLEMP_EXPERIMENT, 3e-2),
    ],
)
def test_rbf_interpolation_coeffs_vertex(
    grid_file, grid_savepoint, interpolation_savepoint, icon_grid, backend, experiment, atol
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.VERTEX

    horizontal_start = icon_grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    assert horizontal_start < grid.num_vertices

    rbf_vec_coeff_v1, rbf_vec_coeff_v2 = rbf.compute_rbf_interpolation_coeffs_vertex(
        geometry.get(geometry_attrs.VERTEX_LAT).ndarray,
        geometry.get(geometry_attrs.VERTEX_LON).ndarray,
        geometry.get(geometry_attrs.VERTEX_X).ndarray,
        geometry.get(geometry_attrs.VERTEX_Y).ndarray,
        geometry.get(geometry_attrs.VERTEX_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid),
        rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        grid.global_properties.geometry_type,
        # TODO(msimberg): Write this to config
        rbf.compute_default_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim)
        if grid.global_properties.geometry_type is base.GeometryType.ICOSAHEDRON
        else grid.mean_dual_edge_length,
        horizontal_start,
        grid.global_properties.domain_length,
        grid.global_properties.domain_height,
        array_ns=data_alloc.import_array_ns(backend),
    )

    rbf_vec_coeff_v1_ref = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_vec_coeff_v2_ref = interpolation_savepoint.rbf_vec_coeff_v2()

    assert rbf_vec_coeff_v1.shape == rbf_vec_coeff_v1_ref.shape
    assert rbf_vec_coeff_v2.shape == rbf_vec_coeff_v2_ref.shape
    assert rbf_vec_coeff_v1_ref.shape == (
        icon_grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert rbf_vec_coeff_v2_ref.shape == (
        icon_grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v1[horizontal_start:],
        rbf_vec_coeff_v1_ref.asnumpy()[horizontal_start:],
        atol=atol,
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v2[horizontal_start:],
        rbf_vec_coeff_v2_ref.asnumpy()[horizontal_start:],
        atol=atol,
    )


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment, atol",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.GLOBAL_EXPERIMENT, 8e-14),
        (dt_utils.REGIONAL_EXPERIMENT, dt_utils.REGIONAL_EXPERIMENT, 2e-9),
        (dt_utils.WEISMAN_KLEMP_EXPERIMENT, dt_utils.WEISMAN_KLEMP_EXPERIMENT, 1e-5),
    ],
)
def test_rbf_interpolation_coeffs_edge(
    grid_file,
    grid_savepoint,
    interpolation_savepoint,
    icon_grid,
    backend,
    experiment,
    atol,
):
    geometry = gridtest_utils.get_grid_geometry(backend, experiment, grid_file)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.EDGE

    horizontal_start = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    assert horizontal_start < grid.num_edges

    rbf_vec_coeff_e = rbf.compute_rbf_interpolation_coeffs_edge(
        geometry.get(geometry_attrs.EDGE_LAT).ndarray,
        geometry.get(geometry_attrs.EDGE_LON).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        geometry.get(geometry_attrs.EDGE_DUAL_U).ndarray,
        geometry.get(geometry_attrs.EDGE_DUAL_V).ndarray,
        # NOTE: Neighbors are not in the same order. Use savepoint to make sure
        # order of coefficients computed by icon4py matches order of
        # coefficients in savepoint.
        grid_savepoint.e2c2e(),
        rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        grid.global_properties.geometry_type,
        # TODO(msimberg): Write this to config
        rbf.compute_default_rbf_scale(math.sqrt(grid_savepoint.mean_cell_area()), rbf_dim)
        if grid.global_properties.geometry_type is base.GeometryType.ICOSAHEDRON
        else grid.mean_dual_edge_length,
        horizontal_start,
        grid.global_properties.domain_length,
        grid.global_properties.domain_height,
        array_ns=data_alloc.import_array_ns(backend),
    )

    rbf_vec_coeff_e_ref = interpolation_savepoint.rbf_vec_coeff_e()

    assert rbf_vec_coeff_e.shape == rbf_vec_coeff_e_ref.shape
    assert rbf_vec_coeff_e_ref.shape == (
        icon_grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_e[horizontal_start:],
        rbf_vec_coeff_e_ref.asnumpy()[horizontal_start:],
        atol=atol,
    )
