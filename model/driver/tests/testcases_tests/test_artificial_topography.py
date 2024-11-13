# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.driver.testcases import artificial_topography
import pytest

from icon4py.model.common.settings import xp
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.test_utils import datatest_utils as dt_utils, helpers

@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.GAUSS3D_EXPERIMENT])
def test_gaussian_hill(
    grid_savepoint,
    constant_fields_savepoint,
    experiment,
    icon_grid,
    backend,
):

    cell_geometry = grid_savepoint.construct_cell_geometry()
    topography_verif = constant_fields_savepoint.topo_c()

    topography = gtx.as_field((dims.CellDim,), xp.zeros((icon_grid.num_cells,), dtype=ta.wpfloat))
    artificial_topography.gaussian_hill.with_backend(backend)(
        cell_center_lon=cell_geometry.cell_center_lon,
        cell_center_lat=cell_geometry.cell_center_lat,
        topography=topography,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        offset_provider={},
    )

    assert helpers.dallclose(
        topography_verif.ndarray, topography.ndarray, atol=1.0e-14
    )