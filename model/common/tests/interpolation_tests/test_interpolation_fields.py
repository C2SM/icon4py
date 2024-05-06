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

from icon4py.model.common.dimension import (
    E2CDim,
    E2VDim,
    EdgeDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import (
    HorizontalMarkerIndex,
)
from icon4py.model.common.interpolation.interpolation_fields import (
    compute_c_lin_e,
    compute_cells_aw_verts,
)
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    download_ser_data,
    experiment,
    processor_props,
    ranked_data_path,
)


@pytest.mark.datatest
def test_compute_c_lin_e(grid_savepoint, interpolation_savepoint, icon_grid):
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    edge_cell_length = grid_savepoint.edge_cell_length()
    owner_mask = grid_savepoint.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()
    lateral_boundary = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    c_lin_e = compute_c_lin_e(
        edge_cell_length.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask.asnumpy(),
        lateral_boundary,
    )

    assert np.allclose(c_lin_e, c_lin_e_ref.asnumpy())


@pytest.mark.datatest
def test_compute_cells_aw_verts(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    dual_area = grid_savepoint.v_dual_area().asnumpy()
    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    owner_mask = grid_savepoint.v_owner_mask()
    e2c = icon_grid.connectivities[E2CDim]
    v2c = icon_grid.connectivities[V2CDim]
    v2e = icon_grid.connectivities[V2EDim]
    e2v = icon_grid.connectivities[E2VDim]
    horizontal_start_vertex = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    horizontal_end_vertex = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )

    cells_aw_verts = compute_cells_aw_verts(
        dual_area,
        edge_vert_length,
        edge_cell_length,
        owner_mask,
        e2c,
        v2c,
        v2e,
        e2v,
        horizontal_start_vertex,
        horizontal_end_vertex,
    )
    cells_aw_verts_ref = interpolation_savepoint.c_intp().asnumpy()
    assert np.allclose(cells_aw_verts, cells_aw_verts_ref)
