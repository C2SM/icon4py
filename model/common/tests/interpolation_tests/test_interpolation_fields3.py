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
    KDim,
)
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.interpolation.interpolation_fields3 import (
    compute_ddxn_z_half_e,
    compute_ddxnt_z_full,
    compute_cells_aw_verts,
    compute_cells2verts_scalar,
    compute_ddxt_z_half_e,
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
def test_compute_ddxn_z_full_e(grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint):  # fixture
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    e2c = icon_grid.connectivities[E2CDim]
#    edge_cell_length = grid_savepoint.edge_cell_length()
#    owner_mask = grid_savepoint.e_owner_mask()
    ddxn_z_full_ref = metrics_savepoint.ddxn_z_full().asnumpy()
    horizontal_start = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    horizontal_end = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = 66
    ddxn_z_half_e = zero_field(icon_grid, EdgeDim, KDim)
    ddxn_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxn_z_half_e(
        z_ifc,
        inv_dual_edge_length,
        out=ddxn_z_half_e,
        offset_provider={"E2C" : icon_grid.get_offset_provider("E2C") },
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
    print(ddxn_z_half_e.asnumpy().shape)
    ddxn_z_full = compute_ddxnt_z_full(
        ddxn_z_half_e.asnumpy(),
    )

    print(ddxn_z_full.shape)
    print(ddxn_z_full_ref.shape)
    assert np.allclose(ddxn_z_full, ddxn_z_full_ref)


@pytest.mark.datatest
def test_compute_ddxt_z_full_e(grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint):
    z_ifc = metrics_savepoint.z_ifc().asnumpy()
    dual_area = grid_savepoint.v_dual_area().asnumpy()
    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    tangent_orientation = grid_savepoint.tangent_orientation().asnumpy()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths().asnumpy()
    owner_mask = grid_savepoint.v_owner_mask()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
    e2c = icon_grid.connectivities[E2CDim]
    v2c = icon_grid.connectivities[V2CDim]
    v2e = icon_grid.connectivities[V2EDim]
    e2v = icon_grid.connectivities[E2VDim]
    second_boundary_layer_start_index_vertex = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    second_boundary_layer_end_index_vertex = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    second_boundary_layer_start_index_cell = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    second_boundary_layer_end_index_cell = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    third_boundary_layer_start_index_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
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
        second_boundary_layer_start_index_vertex,
        second_boundary_layer_end_index_vertex,
    )
    cells_aw_verts_ref = interpolation_savepoint.c_intp().asnumpy()
    assert np.allclose(cells_aw_verts, cells_aw_verts_ref)

    z_ifv = compute_cells2verts_scalar(z_ifc, cells_aw_verts, v2c, second_boundary_layer_start_index_vertex)
    ddxt_z_half_e = compute_ddxt_z_half_e(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
        e2v,
        third_boundary_layer_start_index_edge,
    )
    ddxt_z_full = compute_ddxnt_z_full(
        ddxt_z_half_e,
    )

    assert np.allclose(ddxt_z_full, ddxt_z_full_ref)
