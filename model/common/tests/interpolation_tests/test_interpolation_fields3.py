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
from gt4py.next import as_field

from icon4py.model.common.dimension import (
    E2CDim,
    E2VDim,
    EdgeDim,
    KDim,
    V2CDim,
    V2EDim,
    VertexDim,
)
from icon4py.model.common.grid.horizontal import (
    HorizontalMarkerIndex,
    _compute_cells2verts_scalar,
    compute_cells2edges_scalar,
)
from icon4py.model.common.interpolation.interpolation_fields3 import (
    _compute_ddxnt_z_full,
    compute_cells_aw_verts,
    compute_ddxn_z_half_e,
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
from icon4py.model.common.test_utils.datatest_utils import (
    GLOBAL_EXPERIMENT,
    REGIONAL_EXPERIMENT,
)
from icon4py.model.common.test_utils.helpers import zero_field


@pytest.mark.datatest
def test_compute_ddxn_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    z_ifc = metrics_savepoint.z_ifc()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
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
    vertical_end = icon_grid.num_levels + 1
    ddxn_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxn_z_half_e(
        z_ifc,
        inv_dual_edge_length,
        ddxn_z_half_e,
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    ddxn_z_full = zero_field(icon_grid, EdgeDim, KDim)
    _compute_ddxnt_z_full(
        ddxn_z_half_e,
        out=ddxn_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxn_z_full.asnumpy(), ddxn_z_full_ref)


@pytest.mark.datatest
def test_compute_ddxt_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    z_ifc = metrics_savepoint.z_ifc()
    dual_area = grid_savepoint.v_dual_area().asnumpy()
    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    tangent_orientation = grid_savepoint.tangent_orientation()
    inv_primal_edge_length = grid_savepoint.inverse_primal_edge_lengths()
    owner_mask = grid_savepoint.v_owner_mask()
    ddxt_z_full_ref = metrics_savepoint.ddxt_z_full().asnumpy()
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
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    )
    horizontal_end_edge = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    vertical_start = 0
    vertical_end = icon_grid.num_levels + 1
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

    z_ifv = zero_field(icon_grid, VertexDim, KDim, extend={KDim: 1})
    _compute_cells2verts_scalar(
        z_ifc,
        as_field((VertexDim, V2CDim), cells_aw_verts),
        out=z_ifv,
        offset_provider={"V2C": icon_grid.get_offset_provider("V2C")},
        domain={
            VertexDim: (horizontal_start_vertex, horizontal_end_vertex),
            KDim: (vertical_start, vertical_end),
        },
    )
    ddxt_z_half_e = zero_field(icon_grid, EdgeDim, KDim, extend={KDim: 1})
    compute_ddxt_z_half_e(
        z_ifv,
        inv_primal_edge_length,
        tangent_orientation,
        ddxt_z_half_e,
        horizontal_start_edge,
        horizontal_end_edge,
        vertical_start,
        vertical_end,
        offset_provider={"E2V": icon_grid.get_offset_provider("E2V")},
    )
    ddxt_z_full = zero_field(icon_grid, EdgeDim, KDim)
    _compute_ddxnt_z_full(
        ddxt_z_half_e,
        out=ddxt_z_full,
        offset_provider={"Koff": icon_grid.get_offset_provider("Koff")},
    )

    assert np.allclose(ddxt_z_full.asnumpy(), ddxt_z_full_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", (REGIONAL_EXPERIMENT, GLOBAL_EXPERIMENT))
def test_compute_ddqz_z_full_e(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    c_lin_e = interpolation_savepoint.c_lin_e()
    ddqz_z_full_e_ref = metrics_savepoint.ddqz_z_full_e().asnumpy()
    horizontal_start_edge = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    horizontal_end_edge = icon_grid.get_end_index(  # noqa: F841
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )  # for the global experiment, this outputs 0
    vertical_start = 0
    vertical_end = icon_grid.num_levels
    ddqz_z_full_e = zero_field(icon_grid, EdgeDim, KDim)
    compute_cells2edges_scalar(
        inv_p_cell_in=inv_ddqz_z_full,
        c_int=c_lin_e,
        p_vert_out=ddqz_z_full_e,
        horizontal_start_edge=horizontal_start_edge,
        horizontal_end_edge=ddqz_z_full_e.shape[0],  # horizontal_end_edge,
        vertical_start=vertical_start,
        vertical_end=vertical_end,
        offset_provider={"E2C": icon_grid.get_offset_provider("E2C")},
    )
    assert np.allclose(ddqz_z_full_e.asnumpy(), ddqz_z_full_e_ref)
