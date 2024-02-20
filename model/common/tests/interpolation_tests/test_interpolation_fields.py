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

from icon4py.model.common.dimension import EdgeDim, CellDim, C2EDim, VertexDim, V2EDim, KDim, E2CDim, C2E2CDim, E2VDim, C2VDim, V2CDim, E2C2EODim, E2C2EDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.interpolation.interpolation_fields import compute_c_lin_e, compute_geofac_div, compute_geofac_rot, compute_geofac_n2s, compute_primal_normal_ec, compute_geofac_grg, compute_geofac_grdiv, compute_c_bln_avg, compute_mass_conservation_c_bln_avg, compute_e_flx_avg
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

@pytest.mark.datatest
def test_compute_geofac_grdiv(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    geofac_div = interpolation_savepoint.geofac_div()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
#    geofac_grg = zero_field(icon_grid, CellDim, C2EDim)
    geofac_grdiv_ref = interpolation_savepoint.geofac_grdiv()
    owner_mask = grid_savepoint.c_owner_mask()
    C2E_ = icon_grid.connectivities[C2EDim]
    E2C_ = icon_grid.connectivities[E2CDim]
    C2E2C_ = icon_grid.connectivities[C2E2CDim]
    E2C2E_ = icon_grid.connectivities[E2C2EDim]
    lateral_boundary = np.arange(2)
    lateral_boundary[0] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    geofac_grdiv = np.zeros([lateral_boundary[1], 5])
    geofac_grdiv = compute_geofac_grdiv(
        geofac_grdiv,
        geofac_div.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask,
        C2E_,
        E2C_,
        C2E2C_,
        E2C2E_,
        lateral_boundary,
    )
#    np.set_printoptions(threshold=np.inf)
#    print(geofac_grdiv_ref.asnumpy())
#    print("aaaaa")
#    print(geofac_grdiv)
    assert np.allclose(geofac_grdiv, geofac_grdiv_ref.asnumpy())

# redundant implementation
#@pytest.mark.datatest
#def test_compute_rbf_vec_idx_v(
#    grid_savepoint, interpolation_savepoint, icon_grid
#):
#    num_edges = grid_savepoint.v_num_edges().asnumpy()
#    owner_mask = grid_savepoint.v_owner_mask()
#    rbf_vec_idx_v_ref = interpolation_savepoint.rbf_vec_idx_v().asnumpy()
#    V2E_ = icon_grid.connectivities[V2EDim]
#    lateral_boundary = np.arange(2)
#    lateral_boundary[0] = icon_grid.get_start_index(
#        VertexDim,
#        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
#    )
#    lateral_boundary[1] = icon_grid.get_end_index(
#        VertexDim,
#        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
#    )
#    rbf_vec_idx_v_ref = rbf_vec_idx_v_ref[:, 0:lateral_boundary[1]]
#    rbf_vec_idx_v = compute_rbf_vec_idx_v(V2E_, num_edges, owner_mask, lateral_boundary)
#    assert np.allclose(rbf_vec_idx_v, rbf_vec_idx_v_ref)

@pytest.mark.datatest
def test_compute_c_bln_avg(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    cell_areas = grid_savepoint.cell_areas().asnumpy()
    divavg_cntrwgt = interpolation_savepoint.divavg_cntrwgt().asnumpy()
    c_bln_avg_ref = interpolation_savepoint.c_bln_avg().asnumpy()
    owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    C2E2C = icon_grid.connectivities[C2E2CDim]
    lateral_boundary = np.arange(3)
    lateral_boundary[0] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    lateral_boundary[1] = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    lateral_boundary[2] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
    )
    lat = grid_savepoint.cell_center_lat().asnumpy()
    lon = grid_savepoint.cell_center_lon().asnumpy()
    c_bln_avg = np.zeros([lateral_boundary[1], 4])
    c_bln_avg = compute_c_bln_avg(
        c_bln_avg,
        divavg_cntrwgt,
        owner_mask,
        C2E2C,
        lateral_boundary,
        lat,
        lon,
    )
    c_bln_avg = compute_mass_conservation_c_bln_avg(
        c_bln_avg,
        divavg_cntrwgt,
        owner_mask,
        C2E2C,
        lateral_boundary,
        lat,
        lon,
        cell_areas,
        1000,
    )
#    np.set_printoptions(threshold=np.inf)
    print(c_bln_avg_ref)
    print("aaaaa")
    print(c_bln_avg)
    assert np.allclose(c_bln_avg, c_bln_avg_ref)

@pytest.mark.datatest
def test_compute_e_flx_avg(
    grid_savepoint, interpolation_savepoint, icon_grid
):
    e_flx_avg_ref = interpolation_savepoint.e_flx_avg().asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg().asnumpy()
    geofac_div = interpolation_savepoint.geofac_div().asnumpy()
    owner_mask = grid_savepoint.e_owner_mask().asnumpy()
    primal_cart_normal_x = grid_savepoint.primal_cart_normal_x().asnumpy()
    primal_cart_normal_y = grid_savepoint.primal_cart_normal_y().asnumpy()
    primal_cart_normal_z = grid_savepoint.primal_cart_normal_z().asnumpy()
    primal_cart_normal = np.transpose(np.stack((primal_cart_normal_x, primal_cart_normal_y, primal_cart_normal_z)))
    E2C = icon_grid.connectivities[E2CDim]
    C2E = icon_grid.connectivities[C2EDim]
    C2E2C = icon_grid.connectivities[C2E2CDim]
    E2C2E = icon_grid.connectivities[E2C2EDim]
    lateral_boundary_edges = np.arange(4)
    lateral_boundary_edges[0] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    lateral_boundary_edges[1] = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    lateral_boundary_edges[2] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3,
    )
    lateral_boundary_edges[3] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
    )
    lateral_boundary_cells = np.arange(3)
    lateral_boundary_cells[0] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    lateral_boundary_cells[1] = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    lateral_boundary_cells[2] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
    )
    lat = grid_savepoint.cell_center_lat().asnumpy()
    lon = grid_savepoint.cell_center_lon().asnumpy()
    e_flx_avg = np.zeros([lateral_boundary_edges[1], 5])
    e_flx_avg = compute_e_flx_avg(
        e_flx_avg,
        c_bln_avg,
        geofac_div,
        owner_mask,
        primal_cart_normal,
        E2C,
        C2E,
        C2E2C,
        E2C2E,
        lateral_boundary_cells,
        lateral_boundary_edges,
    )
#    np.set_printoptions(threshold=np.inf)
    print(e_flx_avg_ref)
    print("bbbbb")
    print(e_flx_avg)
    assert np.allclose(e_flx_avg, e_flx_avg_ref)
