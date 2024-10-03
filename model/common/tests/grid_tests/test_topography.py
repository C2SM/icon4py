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