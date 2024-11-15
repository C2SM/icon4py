# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import time

import numpy as np
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid
import icon4py.model.common.test_utils.helpers as test_helpers
from icon4py.model.common import constants
from icon4py.model.common.interpolation.interpolation_fields import (
    compute_c_bln_avg,
    compute_c_lin_e,
    compute_cells_aw_verts,
    compute_e_bln_c_s,
    compute_e_flx_avg,
    compute_force_mass_conservation_to_c_bln_avg,
    compute_geofac_div,
    compute_geofac_grdiv,
    compute_geofac_grg,
    compute_geofac_n2s,
    compute_geofac_rot,
    compute_pos_on_tplane_e_x_y,
    compute_primal_normal_ec,
    create_inverse_neighbor_idx0,
)
from icon4py.model.common.test_utils import datatest_utils as dt_utils
from icon4py.model.common.test_utils.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
    data_provider,
    download_ser_data,
    experiment,
    processor_props,
    ranked_data_path,
)


cell_domain = h_grid.domain(dims.CellDim)
edge_domain = h_grid.domain(dims.EdgeDim)
vertex_domain = h_grid.domain(dims.VertexDim)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_c_lin_e(grid_savepoint, interpolation_savepoint, icon_grid):  # fixture
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    edge_cell_length = grid_savepoint.edge_cell_length()
    owner_mask = grid_savepoint.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()
    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    c_lin_e = compute_c_lin_e(
        edge_cell_length.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask.asnumpy(),
        horizontal_start,
    )

    assert test_helpers.dallclose(c_lin_e, c_lin_e_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_div(grid_savepoint, interpolation_savepoint, icon_grid):
    mesh = icon_grid
    primal_edge_length = grid_savepoint.primal_edge_length()
    edge_orientation = grid_savepoint.edge_orientation()
    area = grid_savepoint.cell_areas()
    geofac_div_ref = interpolation_savepoint.geofac_div()
    geofac_div = test_helpers.zero_field(mesh, dims.CellDim, dims.C2EDim)
    compute_geofac_div(
        primal_edge_length,
        edge_orientation,
        area,
        out=geofac_div,
        offset_provider={"C2E": mesh.get_offset_provider("C2E")},
    )

    assert test_helpers.dallclose(geofac_div.asnumpy(), geofac_div_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_rot(grid_savepoint, interpolation_savepoint, icon_grid):
    mesh = icon_grid
    dual_edge_length = grid_savepoint.dual_edge_length()
    edge_orientation = grid_savepoint.vertex_edge_orientation()
    dual_area = grid_savepoint.vertex_dual_area()
    owner_mask = grid_savepoint.v_owner_mask()
    geofac_rot_ref = interpolation_savepoint.geofac_rot()
    geofac_rot = test_helpers.zero_field(mesh, dims.VertexDim, dims.V2EDim)
    horizontal_start = icon_grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    compute_geofac_rot(
        dual_edge_length,
        edge_orientation,
        dual_area,
        owner_mask,
        out=geofac_rot[horizontal_start:, :],
        offset_provider={"V2E": mesh.get_offset_provider("V2E")},
    )

    assert test_helpers.dallclose(geofac_rot.asnumpy(), geofac_rot_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_n2s(grid_savepoint, interpolation_savepoint, icon_grid):
    dual_edge_length = grid_savepoint.dual_edge_length()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_n2s_ref = interpolation_savepoint.geofac_n2s()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    geofac_n2s = compute_geofac_n2s(
        dual_edge_length.asnumpy(),
        geofac_div.asnumpy(),
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )
    assert test_helpers.dallclose(geofac_n2s, geofac_n2s_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_grg(grid_savepoint, interpolation_savepoint, icon_grid):
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x().asnumpy()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y().asnumpy()
    geofac_div = interpolation_savepoint.geofac_div()
    c_lin_e = interpolation_savepoint.c_lin_e()
    geofac_grg_ref = interpolation_savepoint.geofac_grg()
    owner_mask = grid_savepoint.c_owner_mask()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    primal_normal_ec = compute_primal_normal_ec(
        primal_normal_cell_x,
        primal_normal_cell_y,
        owner_mask,
        c2e,
        e2c,
        horizontal_start,
    )
    geofac_grg = compute_geofac_grg(
        primal_normal_ec,
        geofac_div.asnumpy(),
        c_lin_e.asnumpy(),
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )
    assert test_helpers.dallclose(
        geofac_grg[:, :, 0], geofac_grg_ref[0].asnumpy(), atol=1e-6, rtol=1e-7
    )
    assert test_helpers.dallclose(
        geofac_grg[:, :, 1], geofac_grg_ref[1].asnumpy(), atol=1e-6, rtol=1e-7
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_grdiv(grid_savepoint, interpolation_savepoint, icon_grid):
    geofac_div = interpolation_savepoint.geofac_div()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    geofac_grdiv_ref = interpolation_savepoint.geofac_grdiv()
    owner_mask = grid_savepoint.c_owner_mask()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    e2c2e = icon_grid.connectivities[dims.E2C2EDim]
    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    geofac_grdiv = compute_geofac_grdiv(
        geofac_div.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        owner_mask,
        c2e,
        e2c,
        e2c2e,
        horizontal_start,
    )
    assert test_helpers.dallclose(geofac_grdiv, geofac_grdiv_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_inverse_neighbor(experiment, icon_grid):
    c2e2c0 = icon_grid.connectivities[dims.C2E2CODim]
    inverse_neighbors = create_inverse_neighbor_idx0(c2e2c0)
    #assert np.all(c2e2c0[inverse_neighbors] == x)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, atol",
    [
        (dt_utils.REGIONAL_EXPERIMENT, 1e-10),
        (dt_utils.GLOBAL_EXPERIMENT, 1e-10)
    ],
)
def test_compute_c_bln_avg(grid_savepoint, interpolation_savepoint, icon_grid, experiment, atol):
    cell_areas = grid_savepoint.cell_areas().asnumpy()
    # both experiment use the default value
    divavg_cntrwgt = 0.5
    c_bln_avg_ref = interpolation_savepoint.c_bln_avg().asnumpy()
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    horizontal_start_p2 = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))

    lat = grid_savepoint.cell_center_lat().asnumpy()
    lon = grid_savepoint.cell_center_lon().asnumpy()
    c_bln_avg = compute_c_bln_avg(c2e2c, lat, lon, divavg_cntrwgt, horizontal_start)
    cell_owner_mask = grid_savepoint.c_owner_mask()
    c_bln_avg_v1 = np.copy(c_bln_avg)
    start = time.time()
    c2e2c0 = icon_grid.connectivities[dims.C2E2CODim]
    c_bln_avg_v1 = compute_force_mass_conservation_to_c_bln_avg(c_bln_avg_v1,
                                                                cell_areas,c2e2c0,
                                                                cell_owner_mask,
                                                                divavg_cntrwgt,
                                                                horizontal_start_p2,
                                                                )
    d1 = time.time()-start
    print(f" v1: time {d1}")


    assert test_helpers.dallclose(c_bln_avg_v1, c_bln_avg_ref, atol=atol)  # version 1 of correction AJ
   

@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
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
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e = icon_grid.connectivities[dims.C2EDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    e2c2e = icon_grid.connectivities[dims.E2C2EDim]
    horizontal_start_p3 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    horizontal_start_p4 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
    e_flx_avg = compute_e_flx_avg(
        c_bln_avg,
        geofac_div,
        owner_mask,
        primal_cart_normal,
        e2c,
        c2e,
        c2e2c,
        e2c2e,
        horizontal_start_p3,
        horizontal_start_p4,
    )
    assert test_helpers.dallclose(e_flx_avg, e_flx_avg_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_cells_aw_verts(
    grid_savepoint, interpolation_savepoint, icon_grid, metrics_savepoint
):
    cells_aw_verts_ref = interpolation_savepoint.c_intp().asnumpy()
    dual_area = grid_savepoint.vertex_dual_area().asnumpy()
    edge_vert_length = grid_savepoint.edge_vert_length().asnumpy()
    edge_cell_length = grid_savepoint.edge_cell_length().asnumpy()
    e2c = icon_grid.connectivities[dims.E2CDim]
    v2c = icon_grid.connectivities[dims.V2CDim]
    v2e = icon_grid.connectivities[dims.V2EDim]
    e2v = icon_grid.connectivities[dims.E2VDim]
    horizontal_start_vertex = icon_grid.start_index(
        vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    cells_aw_verts = compute_cells_aw_verts(
        dual_area=dual_area,
        edge_vert_length=edge_vert_length,
        edge_cell_length=edge_cell_length,
        v2e=v2e,
        e2v=e2v,
        v2c=v2c,
        e2c=e2c,
        horizontal_start_vertex=horizontal_start_vertex,
    )
    assert test_helpers.dallclose(cells_aw_verts, cells_aw_verts_ref, atol=1e-3)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_e_bln_c_s(grid_savepoint, interpolation_savepoint, icon_grid):
    e_bln_c_s_ref = interpolation_savepoint.e_bln_c_s().asnumpy()
    c2e = icon_grid.connectivities[dims.C2EDim]
    cells_lat = grid_savepoint.cell_center_lat().asnumpy()
    cells_lon = grid_savepoint.cell_center_lon().asnumpy()
    edges_lat = grid_savepoint.edges_center_lat().asnumpy()
    edges_lon = grid_savepoint.edges_center_lon().asnumpy()
    e_bln_c_s = compute_e_bln_c_s(c2e, cells_lat, cells_lon, edges_lat, edges_lon, 0.0)
    assert test_helpers.dallclose(e_bln_c_s, e_bln_c_s_ref, atol=1e-6, rtol=1e-7)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_pos_on_tplane_e(grid_savepoint, interpolation_savepoint, icon_grid):
    pos_on_tplane_e_x_ref = interpolation_savepoint.pos_on_tplane_e_x().asnumpy()
    pos_on_tplane_e_y_ref = interpolation_savepoint.pos_on_tplane_e_y().asnumpy()
    sphere_radius = constants.EARTH_RADIUS
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
    e2c = icon_grid.connectivities[dims.E2CDim]
    e2v = icon_grid.connectivities[dims.E2VDim]
    e2c2e = icon_grid.connectivities[dims.E2C2EDim]
    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
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
        e2c,
        e2v,
        e2c2e,
        horizontal_start,
    )
    assert test_helpers.dallclose(pos_on_tplane_e_x, pos_on_tplane_e_x_ref, atol=1e-6, rtol=1e-7)
    assert test_helpers.dallclose(pos_on_tplane_e_y, pos_on_tplane_e_y_ref, atol=1e-6, rtol=1e-7)
