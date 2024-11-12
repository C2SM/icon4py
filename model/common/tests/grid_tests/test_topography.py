# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import geometry, topography as topo
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers, reference_funcs


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        (dt_utils.GAUSS3D_EXPERIMENT),
    ],
)
def test_topography_smoothing_withSerializedData(
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    constant_fields_savepoint,
    backend,
    experiment,
):
    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    num_iterations = 25
    topography = constant_fields_savepoint.topo_c()
    topography_smoothed_verif_np = constant_fields_savepoint.topo_smt_c().ndarray

    topography_smoothed = topo.compute_smooth_topo(
        topography=topography,
        grid=icon_grid,
        cell_areas=cell_geometry.area,
        geofac_n2s=geofac_n2s,
        backend=backend,
        num_iterations=num_iterations,
    )

    assert helpers.dallclose(
        topography_smoothed_verif_np, topography_smoothed.ndarray, atol=1.0e-14
    )


@pytest.mark.datatest
def test_topography_smoothing_withNumpy(
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    backend,
):
    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    num_iterations = 2
    topography = helpers.random_field(icon_grid, dims.CellDim, dtype=ta.wpfloat)
    topography_np = topography.ndarray

    # numpy implementation
    topography_smoothed_np = xp.zeros((icon_grid.num_cells, icon_grid.num_levels), dtype=ta.wpfloat)
    topography_smoothed_np[:, 0] = topography_np.copy()
    for _ in range(num_iterations):
        nabla2_topo_np = reference_funcs.nabla2_scalar_numpy(
            icon_grid, topography_smoothed_np, geofac_n2s.ndarray
        )
        topography_smoothed_np[:, 0] = (
            topography_smoothed_np[:, 0] + 0.125 * nabla2_topo_np[:, 0] * cell_geometry.area.ndarray
        )
    topography_smoothed_verif_np = topography_smoothed_np[:, 0]

    topography_smoothed = topo.compute_smooth_topo(
        topography=topography,
        grid=icon_grid,
        cell_areas=cell_geometry.area,
        geofac_n2s=geofac_n2s,
        backend=backend,
        num_iterations=num_iterations,
    )

    assert helpers.dallclose(topography_smoothed_verif_np, topography_smoothed.ndarray)
