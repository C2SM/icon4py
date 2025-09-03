# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from icon4py.model.testing import definitions
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import grid_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common import constants
from icon4py.model.common.grid.geometry_stencils import compute_edge_length, compute_cell_center_arc_distance
from ..fixtures import *

@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, grid_file",
    ((definitions.Experiments.MCH_CH_R04B09.name, definitions.Grids.MCH_CH_R04B09_DSL.name), (definitions.Experiments.EXCLAIM_APE.name, definitions.Grids.R02B04_GLOBAL.name)),
)
def test_edge_length(experiment, grid_file, grid_savepoint, backend):
    keep = True
    #grid = grid_savepoint.construct_icon_grid(backend, keep_skip_values=keep)

    gm = grid_utils.get_grid_manager_from_identifier(grid_file, keep_skip_values=keep, num_levels=1,
                                                             backend=backend)
    grid = gm.grid
    coordinates = gm.coordinates[dims.VertexDim]
    lat = coordinates["lat"]
    lon = coordinates["lon"]
    #lat = grid_savepoint.lat(dims.VertexDim)
    #lon = grid_savepoint.lon(dims.VertexDim)
    length = data_alloc.zero_field(grid, dims.EdgeDim)
    compute_edge_length.with_backend(backend)(
        vertex_lat=lat,
        vertex_lon=lon,
        radius=constants.EARTH_RADIUS,
        length=length,
        horizontal_start=0,
        horizontal_end=grid.size[dims.EdgeDim],
        offset_provider={"E2V": grid.get_connectivity(dims.E2V)},
    )

    assert np.allclose(length.asnumpy(), grid_savepoint.primal_edge_length().asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, grid_file",
    ((definitions.Experiments.MCH_CH_R04B09.name, definitions.Grids.MCH_CH_R04B09_DSL.name), (definitions.Experiments.EXCLAIM_APE.name, definitions.Grids.R02B04_GLOBAL.name)),
)
def test_compute_dual_edge_length(experiment, grid_file, grid_savepoint, backend):
    keep = True
    # grid = grid_savepoint.construct_icon_grid(backend, keep_skip_values=keep)

    gm = grid_utils.get_grid_manager_from_identifier(grid_file, keep_skip_values=keep, num_levels=1,
                                                     backend=backend)
    grid = gm.grid
    coordinates = gm.coordinates[dims.VertexDim]
    lat = coordinates["lat"]
    lon = coordinates["lon"]
    # lat = grid_savepoint.lat(dims.VertexDim)
    # lon = grid_savepoint.lon(dims.VertexDim)
    length = data_alloc.zero_field(grid, dims.EdgeDim)
    compute_cell_center_arc_distance.with_backend(backend)()
