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
from icon4py.model.common.interpolation.interpolation_fields import (
    compute_c_bln_avg,
    compute_c_lin_e,
    compute_cells_aw_verts,
    compute_e_bln_c_s,
    compute_e_flx_avg,
    compute_geofac_div,
    compute_geofac_grdiv,
    compute_geofac_grg,
    compute_geofac_n2s,
    compute_geofac_rot,
    compute_mass_conservation_c_bln_avg,
    compute_pos_on_tplane_e_x_y,
    compute_primal_normal_ec,
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
def test_compute_c_lin_e(grid_savepoint, interpolation_savepoint, icon_grid):  # fixture
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
        area,
        out=geofac_div,
        offset_provider={"C2E": mesh.get_offset_provider("C2E")},
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
    horizontal_start = int32(
        icon_grid.get_start_index(VertexDim, HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1)
    )
    compute_geofac_rot(
        dual_edge_length,
        edge_orientation,
        dual_area,
        owner_mask,
        out=geofac_rot[horizontal_start:, :],
        offset_provider={"V2E": mesh.get_offset_provider("V2E")},
    )

    assert np.allclose(geofac_rot.asnumpy(), geofac_rot_ref.asnumpy())


@pytest.mark.datatest
def test_compute_geofac_n2s(grid_savepoint, interpolation_savepoint, icon_grid):
    dual_edge_length = grid_savepoint.dual_edge_length()
    geofac_div = interpolation_savepoint.geofac_div()
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
    geofac_n2s = compute_geofac_n2s(
        dual_edge_length.asnumpy(),
        geofac_div.asnumpy(),
        C2E_,
        E2C_,
        C2E2C_,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    assert np.allclose(geofac_n2s, geofac_n2s_ref.asnumpy())


@pytest.mark.datatest
def test_compute_geofac_grg(grid_savepoint, interpolation_savepoint, icon_grid):
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x().asnumpy()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y().asnumpy()
    geofac_div = interpolation_savepoint.geofac_div()
    c_lin_e = interpolation_savepoint.c_lin_e()
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
    primal_normal_ec = compute_primal_normal_ec(
        primal_normal_cell_x,
        primal_normal_cell_y,
        owner_mask,
        C2E_,
        E2C_,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    geofac_grg = compute_geofac_grg(
        primal_normal_ec,
        geofac_div.asnumpy(),
        c_lin_e.asnumpy(),
        C2E_,
        E2C_,
        C2E2C_,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    assert np.allclose(geofac_grg[:, :, 0], geofac_grg_ref[0].asnumpy())
    assert np.allclose(geofac_grg[:, :, 1], geofac_grg_ref[1].asnumpy())


@pytest.mark.datatest
def test_compute_geofac_grdiv(grid_savepoint, interpolation_savepoint, icon_grid):
    geofac_div = interpolation_savepoint.geofac_div()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
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
    geofac_grdiv = compute_geofac_grdiv(
        geofac_div.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask,
        C2E_,
        E2C_,
        C2E2C_,
        E2C2E_,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    assert np.allclose(geofac_grdiv, geofac_grdiv_ref.asnumpy())


@pytest.mark.datatest
def test_compute_c_bln_avg(grid_savepoint, interpolation_savepoint, icon_grid):
    cell_areas = grid_savepoint.cell_areas().asnumpy()
    divavg_cntrwgt = 0.5
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
    c_bln_avg = compute_c_bln_avg(
        divavg_cntrwgt,
        owner_mask,
        C2E2C,
        lat,
        lon,
        lateral_boundary[0],
        lateral_boundary[1],
    )
    c_bln_avg = compute_mass_conservation_c_bln_avg(
        c_bln_avg,
        divavg_cntrwgt,
        owner_mask,
        C2E2C,
        lat,
        lon,
        cell_areas,
        1000,
        lateral_boundary[0],
        lateral_boundary[1],
        lateral_boundary[2],
    )
    assert np.allclose(c_bln_avg, c_bln_avg_ref, atol=1e-4, rtol=1e-5)


@pytest.mark.datatest
def test_compute_e_flx_avg(grid_savepoint, interpolation_savepoint, icon_grid):
    e_flx_avg_ref = interpolation_savepoint.e_flx_avg().asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg().asnumpy()
    geofac_div = interpolation_savepoint.geofac_div().asnumpy()
    owner_mask = grid_savepoint.e_owner_mask().asnumpy()
    primal_cart_normal_x = grid_savepoint.primal_cart_normal_x().asnumpy()
    primal_cart_normal_y = grid_savepoint.primal_cart_normal_y().asnumpy()
    primal_cart_normal_z = grid_savepoint.primal_cart_normal_z().asnumpy()
    primal_cart_normal = np.transpose(
        np.stack((primal_cart_normal_x, primal_cart_normal_y, primal_cart_normal_z))
    )
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
    e_flx_avg = compute_e_flx_avg(
        c_bln_avg,
        geofac_div,
        owner_mask,
        primal_cart_normal,
        E2C,
        C2E,
        C2E2C,
        E2C2E,
        lateral_boundary_cells[1],
        lateral_boundary_edges[1],
        lateral_boundary_edges[2],
        lateral_boundary_edges[3],
    )
    assert np.allclose(e_flx_avg, e_flx_avg_ref)


@pytest.mark.datatest
def test_compute_cells_aw_verts(grid_savepoint, interpolation_savepoint, icon_grid):
    cells_aw_verts_ref = interpolation_savepoint.c_intp().asnumpy()
    dual_area = grid_savepoint.vertex_dual_area().asnumpy()
    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    owner_mask = grid_savepoint.v_owner_mask().asnumpy()
    V2E = icon_grid.connectivities[V2EDim]
    E2V = icon_grid.connectivities[E2VDim]
    V2C = icon_grid.connectivities[V2CDim]
    E2C = icon_grid.connectivities[E2CDim]
    lateral_boundary_verts = np.arange(2)
    lateral_boundary_verts[0] = icon_grid.get_start_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    )
    lateral_boundary_verts[1] = icon_grid.get_end_index(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) - 1,
    )
    cells_aw_verts = compute_cells_aw_verts(
        dual_area,
        edge_vert_length,
        edge_cell_length,
        owner_mask,
        V2E,
        E2V,
        V2C,
        E2C,
        lateral_boundary_verts[0],
        lateral_boundary_verts[1],
    )
    assert np.allclose(cells_aw_verts, cells_aw_verts_ref)


@pytest.mark.datatest
def test_compute_e_bln_c_s(grid_savepoint, interpolation_savepoint, icon_grid):
    e_bln_c_s_ref = interpolation_savepoint.e_bln_c_s().asnumpy()
    owner_mask = grid_savepoint.c_owner_mask().asnumpy()
    C2E = icon_grid.connectivities[C2EDim]
    cells_lat = grid_savepoint.cell_center_lat().asnumpy()
    cells_lon = grid_savepoint.cell_center_lon().asnumpy()
    edges_lat = grid_savepoint.edges_center_lat().asnumpy()
    edges_lon = grid_savepoint.edges_center_lon().asnumpy()
    lateral_boundary_cells = np.arange(2)
    lateral_boundary_cells[0] = icon_grid.get_start_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    )
    lateral_boundary_cells[1] = icon_grid.get_end_index(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) - 1,
    )
    e_bln_c_s = compute_e_bln_c_s(
        owner_mask,
        C2E,
        cells_lat,
        cells_lon,
        edges_lat,
        edges_lon,
        lateral_boundary_cells[1],
    )
    assert np.allclose(e_bln_c_s, e_bln_c_s_ref)


@pytest.mark.datatest
def test_compute_pos_on_tplane_e(grid_savepoint, interpolation_savepoint, icon_grid):
    pos_on_tplane_e_x_ref = interpolation_savepoint.pos_on_tplane_e_x().asnumpy()
    pos_on_tplane_e_y_ref = interpolation_savepoint.pos_on_tplane_e_y().asnumpy()
    sphere_radius = grid_savepoint.sphere_radius().asnumpy()
    primal_normal_v1 = grid_savepoint.primal_normal_v1().asnumpy()
    primal_normal_v2 = grid_savepoint.primal_normal_v2().asnumpy()
    dual_normal_v1 = grid_savepoint.dual_normal_v1().asnumpy()
    dual_normal_v2 = grid_savepoint.dual_normal_v2().asnumpy()
    owner_mask = grid_savepoint.e_owner_mask().asnumpy()
    cells_lon = grid_savepoint.cell_center_lon().asnumpy()
    cells_lat = grid_savepoint.cell_center_lat().asnumpy()
    edges_lon = grid_savepoint.edges_center_lon().asnumpy()
    edges_lat = grid_savepoint.edges_center_lat().asnumpy()
    verts_lon = grid_savepoint.verts_vertex_lon().asnumpy()
    verts_lat = grid_savepoint.verts_vertex_lat().asnumpy()
    E2C = icon_grid.connectivities[E2CDim]
    E2V = icon_grid.connectivities[E2VDim]
    E2C2E = icon_grid.connectivities[E2C2EDim]
    lateral_boundary_edges = np.arange(2)
    lateral_boundary_edges[0] = icon_grid.get_start_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    )
    lateral_boundary_edges[1] = icon_grid.get_end_index(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) - 1,
    )
    pos_on_tplane_e_x, pos_on_tplane_e_y = compute_pos_on_tplane_e_x_y(
        sphere_radius,
        primal_normal_v1,
        primal_normal_v2,
        dual_normal_v1,
        dual_normal_v2,
        cells_lon,
        cells_lat,
        edges_lon,
        edges_lat,
        verts_lon,
        verts_lat,
        owner_mask,
        E2C,
        E2V,
        E2C2E,
        lateral_boundary_edges[0],
        lateral_boundary_edges[1],
    )
    assert np.allclose(pos_on_tplane_e_x, pos_on_tplane_e_x_ref)
    assert np.allclose(pos_on_tplane_e_y, pos_on_tplane_e_y_ref)
