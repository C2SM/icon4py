# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common.grid import geometry, topography as topo
from icon4py.model.testing import definitions as test_definitions, helpers


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment",
    [
        (test_definitions.Experiment.GAUSS3D),
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
    topography_smoothed_verif_np = topography_savepoint.topo_smt_c().asnumpy()

    topography_smoothed = topo.smooth_topography(
        topography=topography,
        grid=icon_grid,
        cell_areas=cell_geometry.area,
        geofac_n2s=geofac_n2s,
        backend=backend,
        num_iterations=num_iterations,
    )

    assert helpers.dallclose(
        topography_smoothed_verif_np, topography_smoothed.asnumpy(), atol=1.0e-14
    )
