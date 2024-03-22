# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest
from gt4py.next.iterator.builtins import int32

from icon4py.model.common.dimension import (
    C2E2CDim,
    C2EDim,
    CellDim,
    E2C2EDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.interpolation_fields3 import (
    compute_ddxn_z_half_e,
    compute_ddxnt_z_full,
)
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    datapath,
    download_ser_data,
    experiment,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import zero_field


@pytest.mark.datatest
def test_compute_ddxn_z_half_e(grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint):  # fixture
    z_ifc = metrics_savepoint.z_ifc().asnumpy()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length().asnumpy()
    e2c = icon_grid.connectivities[E2CDim]
#    edge_cell_length = grid_savepoint.edge_cell_length()
#    owner_mask = grid_savepoint.e_owner_mask()
    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full().asnumpy()
    lateral_boundary = np.arange(2)
    lateral_boundary[0] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    ddxn_z_half_e = compute_ddxn_z_half_e(
        z_ifc,
        inv_dual_edge_length,
        e2c,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    ddxn_z_full = compute_ddxnt_z_full(
        ddxn_z_half_e,
    )

    assert np.allclose(ddxn_z_full, ddxn_z_full_ref)


@pytest.mark.datatest
def test_compute_ddxt_z_half_e(grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint):
#    dual_area = grid_savepoint.dual_area().asnumpy()
#    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    cells_aw_verts = compute_cells_aw_verts(
        dual_area,
        edge_vert_length,
        edge_cell_length,
    )
