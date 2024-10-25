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
from icon4py.model.common.test_utils import helpers as test_helpers, reference_funcs


@pytest.mark.datatest
def test_topography_smoothing(
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    backend,
):
    num_iterations = 2
    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    topography = test_helpers.random_field(icon_grid, dims.CellDim, dtype=ta.wpfloat)
    topography_np = topography.asnumpy()

    # numpy implementation
    topography_smoothed_np = xp.zeros((icon_grid.num_cells, icon_grid.num_levels), dtype=ta.wpfloat)
    topography_smoothed_np[:, 0] = topography_np.copy()
    for iter in range(num_iterations):
        nabla2_topo_np = reference_funcs.nabla2_scalar_numpy(
            icon_grid, topography_smoothed_np, geofac_n2s.asnumpy()
        )
        topography_smoothed_np[:, 0] = (
            topography_smoothed_np[:, 0]
            + 0.125 * nabla2_topo_np[:, 0] * cell_geometry.area.asnumpy()
        )
    topography_smoothed_np = topography_smoothed_np[:, 0]

    # GT4Py implementation
    topography_smoothed = topo.compute_smooth_topo(
        topography=topography,
        grid=icon_grid,
        cell_areas=cell_geometry.area,
        geofac_n2s=geofac_n2s,
        backend=backend,
        num_iterations=num_iterations,
    )

    assert test_helpers.dallclose(topography_smoothed_np, topography_smoothed.asnumpy())
