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

from icon4py.model.common.dimension import EdgeDim, CellDim, C2EDim, VertexDim, V2EDim, KDim, E2CDim, C2E2CDim, E2VDim, C2VDim, V2CDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.interpolation.interpolation_fields import compute_c_lin_e, compute_geofac_div, compute_geofac_rot, compute_geofac_n2s, compute_primal_normal_ec, compute_geofac_grg, compute_rbf_vec_idx_v
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    datapath,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    processor_props,
    ranked_data_path,
)
from icon4py.model.common.test_utils.helpers import zero_field, as_1D_sparse_field, random_field
from icon4py.model.common.grid.simple import SimpleGrid
from gt4py.next.iterator.builtins import int32


@pytest.mark.datatest
def test_compute_c_lin_e(
    grid_savepoint, interpolation_savepoint, icon_grid  # noqa: F811  # fixture
):
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
def test_compute_geofac_div(grid_savepoint, interpolation_savepoint, icon_grid):
    mesh = icon_grid
    primal_edge_length = grid_savepoint.primal_edge_length()
    edge_orientation = grid_savepoint.edge_orientation()
    area = grid_savepoint.cell_areas()
    geofac_div_ref = interpolation_savepoint.geofac_div()
    geofac_div = zero_field(mesh, CellDim, C2EDim)
    compute_geofac_div(
        primal_edge_length,
        edge_orientation,
        area, out=geofac_div, offset_provider={"C2E": mesh.get_offset_provider("C2E")}
    )

    assert np.allclose(geofac_div.asnumpy(), geofac_div_ref.asnumpy())

@pytest.mark.datatest
def test_compute_geofac_rot(grid_savepoint, interpolation_savepoint, icon_grid):
    mesh = icon_grid
    dual_edge_length = grid_savepoint.dual_edge_length()
    edge_orientation = grid_savepoint.vertex_edge_orientation()
    dual_area = grid_savepoint.vertex_dual_area()
    owner_mask = grid_savepoint.v_owner_mask()
    geofac_rot_ref = interpolation_savepoint.geofac_rot()
    geofac_rot = zero_field(mesh, VertexDim, V2EDim)
    horizontal_start = int32(icon_grid.get_start_index(VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1))
    compute_geofac_rot(
        dual_edge_length,
        edge_orientation,
        dual_area,
        owner_mask,
        out=geofac_rot[horizontal_start:, :],
        offset_provider={"V2E": mesh.get_offset_provider("V2E")}
    )

    assert np.allclose(geofac_rot.asnumpy(), geofac_rot_ref.asnumpy())

@pytest.mark.datatest
def test_compute_geofac_n2s(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    dual_edge_length = grid_savepoint.dual_edge_length()
    geofac_div = interpolation_savepoint.geofac_div()
#    geofac_n2s = zero_field(icon_grid, CellDim, C2EDim)
    geofac_n2s_ref = interpolation_savepoint.geofac_n2s()
    C2E_ = icon_grid.connectivities[C2EDim]
    E2C_ = icon_grid.connectivities[E2CDim]
    C2E2C_ = icon_grid.connectivities[C2E2CDim]
    lateral_boundary = np.arange(2)
    lateral_boundary[0] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    geofac_n2s = np.zeros([lateral_boundary[1], 4])
    geofac_n2s = compute_geofac_n2s(
        geofac_n2s,
        dual_edge_length.asnumpy(),
        geofac_div.asnumpy(),
        C2E_,
        E2C_,
        C2E2C_,
        lateral_boundary,
#        grid_savepoint, interpolation_savepoint, icon_grid,
    )
    assert np.allclose(geofac_n2s, geofac_n2s_ref.asnumpy())

@pytest.mark.datatest
def test_compute_geofac_grg(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x().asnumpy()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y().asnumpy()
    geofac_div = interpolation_savepoint.geofac_div()
    c_lin_e = interpolation_savepoint.c_lin_e()
#    geofac_grg = zero_field(icon_grid, CellDim, C2EDim)
    geofac_grg_ref = interpolation_savepoint.geofac_grg()
    owner_mask = grid_savepoint.c_owner_mask()
    C2E_ = icon_grid.connectivities[C2EDim]
    E2C_ = icon_grid.connectivities[E2CDim]
    C2E2C_ = icon_grid.connectivities[C2E2CDim]
    lateral_boundary = np.arange(2)
    lateral_boundary[0] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    geofac_grg = np.zeros([lateral_boundary[1], 4, 2])
    primal_normal_ec = np.zeros([lateral_boundary[1], 3, 2])
    primal_normal_ec = compute_primal_normal_ec(
        primal_normal_ec,
        primal_normal_cell_x,
        primal_normal_cell_y,
        owner_mask,
        C2E_,
        E2C_,
        lateral_boundary,
    )
    geofac_grg = compute_geofac_grg(
        geofac_grg,
        primal_normal_ec,
        geofac_div.asnumpy(),
        c_lin_e.asnumpy(),
        C2E_,
        E2C_,
        C2E2C_,
        lateral_boundary,
    )
    assert np.allclose(geofac_grg[:, :, 0], geofac_grg_ref[0].asnumpy())
    assert np.allclose(geofac_grg[:, :, 1], geofac_grg_ref[1].asnumpy())

#@pytest.mark.datatest
#def test_compute_geofac_grdiv(
#    grid_savepoint, interpolation_savepoint, icon_grid
#):
#    geofac_div = interpolation_savepoint.geofac_div()
#    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
##    geofac_grg = zero_field(icon_grid, CellDim, C2EDim)
#    geofac_grdiv_ref = interpolation_savepoint.geofac_grdiv()
##    np.set_printoptions(threshold=np.inf)
##    print(geofac_grdiv_ref.asnumpy())
##    aaaaa
#    owner_mask = grid_savepoint.c_owner_mask()
#    C2E_ = icon_grid.connectivities[C2EDim]
#    E2C_ = icon_grid.connectivities[E2CDim]
#    C2E2C_ = icon_grid.connectivities[C2E2CDim]
#    lateral_boundary = np.arange(2)
#    lateral_boundary[0] = icon_grid.get_start_index(
#        CellDim,
#        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
#    )
#    lateral_boundary[1] = icon_grid.get_end_index(
#        CellDim,
#        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
#    )
#    geofac_grdiv = np.zeros([lateral_boundary[1], 4, 2])
#    geofac_grdiv = compute_geofac_grg(
#        geofac_grdiv,
#        geofac_div.asnumpy(),
#        inv_dual_edge_length.asnumpy(),
#        owner_mask,
#        C2E_,
#        E2C_,
#        C2E2C_,
#        lateral_boundary,
#    )
##    np.set_printoptions(threshold=np.inf)
##    print(geofac_grg_ref[0].asnumpy())
##    print("aaaaa")
##    print(geofac_grg[:, :, 0])
#    assert np.allclose(geofac_grdiv[:, :, 0], geofac_grdiv_ref[0].asnumpy())
#    assert np.allclose(geofac_grdiv[:, :, 1], geofac_grdiv_ref[1].asnumpy())

@pytest.mark.datatest
def test_compute_rbf_vec_idx_v(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    num_edges = grid_savepoint.v_num_edges().asnumpy()
    owner_mask = grid_savepoint.v_owner_mask()
    rbf_vec_idx_v_ref = interpolation_savepoint.rbf_vec_idx_v().asnumpy()
    V2E_ = icon_grid.connectivities[V2EDim]
    lateral_boundary = np.arange(2)
    lateral_boundary[0] = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    rbf_vec_idx_v_ref = rbf_vec_idx_v_ref[:, 0:lateral_boundary[1]]
    rbf_vec_idx_v = compute_rbf_vec_idx_v(V2E_, num_edges, owner_mask, lateral_boundary)
    assert np.allclose(rbf_vec_idx_v, rbf_vec_idx_v_ref)
