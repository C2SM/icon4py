# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid.geometry_stencils import compute_edge_length
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, grid_utils, serialbox

from ..fixtures import backend, experiment, grid_savepoint, data_provider, download_ser_data, processor_props, ranked_data_path


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_edge_length(
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    keep = True
    grid_file = experiment.grid
    gm = grid_utils.get_grid_manager_from_identifier(
        grid_file, keep_skip_values=keep, num_levels=1, allocator=backend
    )
    grid = gm.grid
    coordinates = gm.coordinates[dims.VertexDim]
    lat = coordinates["lat"]
    lon = coordinates["lon"]
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
