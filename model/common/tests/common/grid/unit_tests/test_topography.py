# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.grid import geometry, topography as topo
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils, test_utils


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        dt_utils.GAUSS3D_EXPERIMENT,
        dt_utils.REGIONAL_EXPERIMENT,
    ],
)
def test_topography_smoothing_with_serialized_data(
    icon_grid,
    grid_savepoint,
    interpolation_savepoint,
    topography_savepoint,
    backend,
    experiment,
):
    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    num_iterations = 25
    topography = topography_savepoint.topo_c()
    xp = data_alloc.import_array_ns(backend)
    topography_smoothed_ref = topography_savepoint.topo_smt_c().asnumpy()

    topography_smoothed = topo.smooth_topography(
        topography=topography.ndarray,
        cell_areas=cell_geometry.area.ndarray,
        geofac_n2s=geofac_n2s.ndarray,
        c2e2co=icon_grid.get_connectivity("C2E2CO").ndarray,
        num_iterations=num_iterations,
        array_ns=xp,
    )

    assert test_utils.dallclose(topography_smoothed_ref, topography_smoothed, atol=1.0e-14)
