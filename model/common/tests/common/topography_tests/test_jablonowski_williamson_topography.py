# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from icon4py.model.common.initialization.jablonowski_williamson_topography import (
    jablonowski_williamson_topography,
)

from icon4py.model.testing import datatest_utils as dt_utils, grid_utils
from icon4py.model.testing import test_utils
from icon4py.model.common.grid import geometry as grid_geometry
import icon4py.model.common.grid.states as grid_states
from icon4py.model.common.grid import geometry_attributes as geometry_meta
from icon4py.model.testing.fixtures.stencil_tests import construct_dummy_decomposition_info
from model.common.tests.common.fixtures import *  # noqa: F403


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_file, experiment",
    [
        (dt_utils.R02B04_GLOBAL, dt_utils.JABW_EXPERIMENT),
    ],
)
def test_jablonowski_williamson_topography(
    grid_file,
    backend,
    topography_savepoint,
):
    num_levels = 65
    grid_manager = grid_utils.get_grid_manager_from_identifier(
        grid_file_identifier=grid_file,
        num_levels=num_levels,
        keep_skip_values=True,
        backend=backend,
    )
    grid = grid_manager.grid
    coordinates = grid_manager.coordinates
    geometry_input_fields = grid_manager.geometry_fields

    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid,
        decomposition_info=construct_dummy_decomposition_info(grid, backend),
        backend=backend,
        coordinates=coordinates,
        extra_fields=geometry_input_fields,
        metadata=geometry_meta.attrs,
    )

    cell_geometry = grid_states.CellParams(
        cell_center_lat=geometry_field_source.get(geometry_meta.CELL_LAT),
        cell_center_lon=geometry_field_source.get(geometry_meta.CELL_LON),
        area=geometry_field_source.get(geometry_meta.CELL_AREA),
    )

    topo_c = jablonowski_williamson_topography(
        cell_lat=cell_geometry.cell_center_lat.asnumpy(),
        u0=35.0,
        backend=backend,
    )

    topo_c_ref = topography_savepoint.topo_c().asnumpy()

    assert test_utils.dallclose(
        topo_c,
        topo_c_ref,
    )
