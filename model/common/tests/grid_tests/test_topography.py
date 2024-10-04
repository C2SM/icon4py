# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import pytest

from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.grid import geometry, icon as icon_grid
from icon4py.model.common.grid import topography as topo
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.interpolation import interpolation_fields as interp_fields

from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils import helpers as test_helpers
from icon4py.model.common.settings import xp

from icon4py.model.common.test_utils.helpers import StencilTest, constant_field, zero_field

def nabla2_scalar_numpy(grid, psi_c: xp.array, geofac_n2s: xp.array):
    c2e2cO = grid.connectivities[dims.C2E2CODim]
    geofac_n2s = xp.expand_dims(geofac_n2s, axis=-1)
    nabla2_psi_c = xp.sum(
        xp.where((c2e2cO != -1)[:, :, xp.newaxis], psi_c[c2e2cO] * geofac_n2s, 0), axis=1
    )
    return nabla2_psi_c


@pytest.mark.datatest
class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = topo.nabla2_scalar
    OUTPUTS = ("nabla2_psi_c",)

    @staticmethod
    def reference(grid, psi_c: xp.array, geofac_n2s: xp.array, **kwargs) -> dict:
        nabla2_psi_c = nabla2_scalar_numpy(grid, psi_c, geofac_n2s)
        return dict(nabla2_psi_c=nabla2_psi_c)

    @pytest.fixture
    def input_data(self, grid, interpolation_savepoint):
        psi_c = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        #geofac_n2s = interpolation_savepoint.geofac_n2s()
        nabla2_psi_c = zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
        )

@pytest.mark.datatest
## @pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_nabla2_scalar(
    grid,
    interpolation_savepoint,
):
        psi_c = constant_field(grid, 1.0, dims.CellDim, dims.KDim)
        geofac_n2s = constant_field(grid, 2.0, dims.CellDim, dims.C2E2CODim)
        #geofac_n2s = interpolation_savepoint.geofac_n2s()
        nabla2_psi_c = zero_field(grid, dims.CellDim, dims.KDim)

        topo.nabla2_scalar(
            psi_c=psi_c,
            geofac_n2s=geofac_n2s,
            nabla2_psi_c=nabla2_psi_c,
            offset_provider={"C2E2CO":grid.get_offset_provider("C2E2CO"),}
        )

        nabla2_psi_c_np = nabla2_scalar_numpy(grid, psi_c.asnumpy(), geofac_n2s.asnumpy())

        assert test_helpers.dallclose(nabla2_psi_c.asnumpy(), nabla2_psi_c_np)


@pytest.mark.datatest
## @pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_topography_smoothing(
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
):

    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    cell_domain = h_grid.domain(dims.CellDim)

    dual_edge_length = grid_savepoint.dual_edge_length() # TODO (Jacopo): add dual_edge_length to torus serialized data
    geofac_div = interpolation_savepoint.geofac_div()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    geofac_n2s_np = interp_fields.compute_geofac_n2s(
        dual_edge_length.asnumpy(),
        geofac_div.asnumpy(),
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )
    geofac_n2s = gtx.as_field((dims.CellDim, dims.C2E2CODim), geofac_n2s_np)
    #geofac_n2s = interpolation_savepoint.geofac_n2s()

    topography_np = xp.zeros((icon_grid.num_cells), dtype=ta.wpfloat)
    topography = gtx.as_field((dims.CellDim,), topography_np)

    topography_smoothed = topo.compute_smooth_topo(
        topography=topography,
        grid=icon_grid,
        cell_areas=cell_geometry.area,
        geofac_n2s=geofac_n2s,
        num_iterations=25,
    )

    assert test_helpers.dallclose(topography.asnumpy(), topography_smoothed.asnumpy())