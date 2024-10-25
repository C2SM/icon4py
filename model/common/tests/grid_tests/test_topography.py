# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.grid import geometry, topography as topo
from icon4py.model.common import dimension as dims, type_alias as ta

from icon4py.model.common.test_utils import helpers as test_helpers
from icon4py.model.common.settings import xp

from icon4py.model.common.test_utils.helpers import constant_field, zero_field

def nabla2_scalar_numpy(grid, psi_c: xp.array, geofac_n2s: xp.array):
    c2e2cO = grid.connectivities[dims.C2E2CODim]
    geofac_n2s = xp.expand_dims(geofac_n2s, axis=-1)
    nabla2_psi_c = xp.sum(
        xp.where((c2e2cO != -1)[:, :, xp.newaxis], psi_c[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return nabla2_psi_c

@pytest.mark.datatest
def test_nabla2_scalar(
    icon_grid,
    interpolation_savepoint,
    backend,
):
        psi_c = constant_field(icon_grid, 1.0, dims.CellDim, dims.KDim)
        #h = constant_field(icon_grid, 2.0, dims.CellDim, dims.C2E2CODim)
        geofac_n2s = interpolation_savepoint.geofac_n2s()
        nabla2_psi_c = zero_field(icon_grid, dims.CellDim, dims.KDim)

        topo.nabla2_scalar.with_backend(backend)(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            offset_provider={"C2E2CO":icon_grid.get_offset_provider("C2E2CO"),}
        )

        nabla2_psi_c_np = nabla2_scalar_numpy(icon_grid, psi_c.asnumpy(), geofac_n2s.asnumpy())

        assert test_helpers.dallclose(nabla2_psi_c.asnumpy(), nabla2_psi_c_np)


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
    topography_smoothed_np[:,0] = topography_np.copy()
    for iter in range(num_iterations):
        nabla2_topo_np = nabla2_scalar_numpy(icon_grid, topography_smoothed_np, geofac_n2s.asnumpy())
        topography_smoothed_np[:,0] = topography_smoothed_np[:,0] + 0.125 * nabla2_topo_np[:,0] * cell_geometry.area.asnumpy()
    topography_smoothed_np = topography_smoothed_np[:,0]

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