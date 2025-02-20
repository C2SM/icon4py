# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools

import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid
import icon4py.model.testing.helpers as test_helpers
from icon4py.model.common import constants
from icon4py.model.common.interpolation.interpolation_fields import (
    compute_c_lin_e,
    compute_cells_aw_verts,
    compute_e_bln_c_s,
    compute_e_flx_avg,
    compute_geofac_div,
    compute_geofac_grdiv,
    compute_geofac_grg,
    compute_geofac_n2s,
    compute_geofac_rot,
    compute_mass_conserving_bilinear_cell_average_weight,
    compute_pos_on_tplane_e_x_y,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import datatest_utils as dt_utils
from icon4py.model.testing.datatest_fixtures import (  # noqa: F401  # import fixtures from test_utils package
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
def test_compute_c_lin_e(grid_savepoint, interpolation_savepoint, icon_grid, backend):  # fixture
    xp = data_alloc.import_array_ns(backend)
    func = functools.partial(compute_c_lin_e, array_ns=xp)
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    edge_cell_length = grid_savepoint.edge_cell_length()
    edge_owner_mask = grid_savepoint.e_owner_mask()
    c_lin_e_ref = interpolation_savepoint.c_lin_e()

    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    c_lin_e = compute_c_lin_e(
        edge_cell_length.asnumpy(),
        inv_dual_edge_length.asnumpy(),
        edge_owner_mask.asnumpy(),
        horizontal_start,
        xp,
    )
    assert test_helpers.dallclose(c_lin_e, c_lin_e_ref.asnumpy())

    c_lin_e_partial = func(
        edge_cell_length.ndarray,
        inv_dual_edge_length.ndarray,
        edge_owner_mask.ndarray,
        horizontal_start,
    )
    assert test_helpers.dallclose(data_alloc.as_numpy(c_lin_e_partial), c_lin_e_ref.asnumpy())


@pytest.mark.embedded_only
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_div(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    mesh = icon_grid
    primal_edge_length = grid_savepoint.primal_edge_length()
    edge_orientation = grid_savepoint.edge_orientation()
    area = grid_savepoint.cell_areas()
    geofac_div_ref = interpolation_savepoint.geofac_div()
    geofac_div = data_alloc.zero_field(mesh, dims.CellDim, dims.C2EDim)
    compute_geofac_div.with_backend(backend)(
        primal_edge_length=primal_edge_length,
        edge_orientation=edge_orientation,
        area=area,
        out=geofac_div,
        offset_provider={"C2E": mesh.get_offset_provider("C2E")},
    )
    assert test_helpers.dallclose(geofac_div.asnumpy(), geofac_div_ref.asnumpy())


@pytest.mark.embedded_only
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_rot(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    mesh = icon_grid
    dual_edge_length = grid_savepoint.dual_edge_length()
    edge_orientation = grid_savepoint.vertex_edge_orientation()
    dual_area = grid_savepoint.vertex_dual_area()
    owner_mask = grid_savepoint.v_owner_mask()
    geofac_rot_ref = interpolation_savepoint.geofac_rot()
    geofac_rot = data_alloc.zero_field(mesh, dims.VertexDim, dims.V2EDim)
    horizontal_start = icon_grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    compute_geofac_rot.with_backend(backend)(
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
def test_compute_geofac_n2s(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    dual_edge_length = grid_savepoint.dual_edge_length()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_n2s_ref = interpolation_savepoint.geofac_n2s()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    geofac_n2s = functools.partial(compute_geofac_n2s, array_ns=xp)(
        dual_edge_length.ndarray,
        geofac_div.ndarray,
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )
    assert test_helpers.dallclose(data_alloc.as_numpy(geofac_n2s), geofac_n2s_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_grg(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x().ndarray
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y().ndarray
    geofac_div = interpolation_savepoint.geofac_div().ndarray
    c_lin_e = interpolation_savepoint.c_lin_e().ndarray
    geofac_grg_ref = interpolation_savepoint.geofac_grg()
    owner_mask = grid_savepoint.c_owner_mask().ndarray
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))

    geofac_grg_0, geofac_grg_1 = functools.partial(compute_geofac_grg, array_ns=xp)(
        primal_normal_cell_x,
        primal_normal_cell_y,
        owner_mask,
        geofac_div,
        c_lin_e,
        c2e,
        e2c,
        c2e2c,
        horizontal_start,
    )
    assert test_helpers.dallclose(
        data_alloc.as_numpy(geofac_grg_0),
        geofac_grg_ref[0].asnumpy(),
        rtol=1e-11,
        atol=1e-19,
    )
    assert test_helpers.dallclose(
        data_alloc.as_numpy(geofac_grg_1),
        geofac_grg_ref[1].asnumpy(),
        rtol=1e-11,
        atol=1e-19,
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_geofac_grdiv(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    geofac_div = interpolation_savepoint.geofac_div()
    inv_dual_edge_length = grid_savepoint.inv_dual_edge_length()
    geofac_grdiv_ref = interpolation_savepoint.geofac_grdiv()
    owner_mask = grid_savepoint.e_owner_mask()
    c2e = icon_grid.connectivities[dims.C2EDim]
    e2c = icon_grid.connectivities[dims.E2CDim]
    e2c2e = icon_grid.connectivities[dims.E2C2EDim]
    horizontal_start = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    geofac_grdiv = functools.partial(compute_geofac_grdiv, array_ns=xp)(
        geofac_div.ndarray,
        inv_dual_edge_length.ndarray,
        owner_mask.ndarray,
        c2e,
        e2c,
        e2c2e,
        horizontal_start,
    )
    assert test_helpers.dallclose(geofac_grdiv, geofac_grdiv_ref.asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, atol",
    [(dt_utils.REGIONAL_EXPERIMENT, 1e-10), (dt_utils.GLOBAL_EXPERIMENT, 1e-10)],
)
def test_compute_c_bln_avg(grid_savepoint, interpolation_savepoint, icon_grid, atol, backend):
    xp = data_alloc.import_array_ns(backend)
    cell_areas = grid_savepoint.cell_areas().ndarray
    # both experiment use the default value
    divavg_cntrwgt = 0.5
    c_bln_avg_ref = interpolation_savepoint.c_bln_avg().asnumpy()
    horizontal_start = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
    horizontal_start_p2 = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3))

    lat = grid_savepoint.cell_center_lat().ndarray
    lon = grid_savepoint.cell_center_lon().ndarray
    cell_owner_mask = grid_savepoint.c_owner_mask().ndarray

    c2e2c0 = icon_grid.connectivities[dims.C2E2CODim]

    c_bln_avg = functools.partial(
        compute_mass_conserving_bilinear_cell_average_weight, array_ns=xp
    )(
        c2e2c0,
        lat,
        lon,
        cell_areas,
        cell_owner_mask,
        divavg_cntrwgt,
        horizontal_start,
        horizontal_start_p2,
    )
    assert test_helpers.dallclose(data_alloc.as_numpy(c_bln_avg), c_bln_avg_ref, atol=atol)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_e_flx_avg(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    e_flx_avg_ref = interpolation_savepoint.e_flx_avg().asnumpy()
    c_bln_avg = interpolation_savepoint.c_bln_avg().ndarray
    geofac_div = interpolation_savepoint.geofac_div().ndarray
    owner_mask = grid_savepoint.e_owner_mask().ndarray
    primal_cart_normal_x = grid_savepoint.primal_cart_normal_x().ndarray
    primal_cart_normal_y = grid_savepoint.primal_cart_normal_y().ndarray
    primal_cart_normal_z = grid_savepoint.primal_cart_normal_z().ndarray
    e2c = icon_grid.connectivities[dims.E2CDim]
    c2e = icon_grid.connectivities[dims.C2EDim]
    c2e2c = icon_grid.connectivities[dims.C2E2CDim]
    e2c2e = icon_grid.connectivities[dims.E2C2EDim]
    horizontal_start_1 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4))
    horizontal_start_2 = icon_grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))

    e_flx_avg = functools.partial(compute_e_flx_avg, array_ns=xp)(
        c_bln_avg,
        geofac_div,
        owner_mask,
        primal_cart_normal_x,
        primal_cart_normal_y,
        primal_cart_normal_z,
        e2c,
        c2e,
        c2e2c,
        e2c2e,
        horizontal_start_1,
        horizontal_start_2,
    )
    assert test_helpers.dallclose(data_alloc.as_numpy(e_flx_avg), e_flx_avg_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_cells_aw_verts(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    cells_aw_verts_ref = interpolation_savepoint.c_intp().asnumpy()
    dual_area = grid_savepoint.vertex_dual_area().ndarray
    edge_vert_length = grid_savepoint.edge_vert_length().ndarray
    edge_cell_length = grid_savepoint.edge_cell_length().ndarray
    e2c = icon_grid.connectivities[dims.E2CDim]
    v2c = icon_grid.connectivities[dims.V2CDim]
    v2e = icon_grid.connectivities[dims.V2EDim]
    e2v = icon_grid.connectivities[dims.E2VDim]
    horizontal_start_vertex = icon_grid.start_index(
        vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    cells_aw_verts = functools.partial(compute_cells_aw_verts, array_ns=xp)(
        dual_area=dual_area,
        edge_vert_length=edge_vert_length,
        edge_cell_length=edge_cell_length,
        v2e=v2e,
        e2v=e2v,
        v2c=v2c,
        e2c=e2c,
        horizontal_start=horizontal_start_vertex,
    )
    assert test_helpers.dallclose(data_alloc.as_numpy(cells_aw_verts), cells_aw_verts_ref)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_e_bln_c_s(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    e_bln_c_s_ref = interpolation_savepoint.e_bln_c_s()
    c2e = icon_grid.connectivities[dims.C2EDim]
    cells_lat = grid_savepoint.cell_center_lat().ndarray
    cells_lon = grid_savepoint.cell_center_lon().ndarray
    edges_lat = grid_savepoint.edges_center_lat().ndarray
    edges_lon = grid_savepoint.edges_center_lon().ndarray
    xp = data_alloc.import_array_ns(backend)

    e_bln_c_s = functools.partial(compute_e_bln_c_s, array_ns=xp)(
        c2e, cells_lat, cells_lon, edges_lat, edges_lon, 0.0
    )
    assert test_helpers.dallclose(
        data_alloc.as_numpy(e_bln_c_s), e_bln_c_s_ref.asnumpy(), atol=1e-6, rtol=1e-7
    )


@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT, dt_utils.GLOBAL_EXPERIMENT])
def test_compute_pos_on_tplane_e(grid_savepoint, interpolation_savepoint, icon_grid, backend):
    xp = data_alloc.import_array_ns(backend)
    pos_on_tplane_e_x_ref = interpolation_savepoint.pos_on_tplane_e_x().asnumpy()
    pos_on_tplane_e_y_ref = interpolation_savepoint.pos_on_tplane_e_y().asnumpy()
    sphere_radius = constants.EARTH_RADIUS
    primal_normal_v1 = grid_savepoint.primal_normal_v1().ndarray
    primal_normal_v2 = grid_savepoint.primal_normal_v2().ndarray
    dual_normal_v1 = grid_savepoint.dual_normal_v1().ndarray
    dual_normal_v2 = grid_savepoint.dual_normal_v2().ndarray
    owner_mask = grid_savepoint.e_owner_mask().ndarray
    cells_lon = grid_savepoint.cell_center_lon().ndarray
    cells_lat = grid_savepoint.cell_center_lat().ndarray
    edges_lon = grid_savepoint.edges_center_lon().ndarray
    edges_lat = grid_savepoint.edges_center_lat().ndarray
    verts_lon = grid_savepoint.verts_vertex_lon().ndarray
    verts_lat = grid_savepoint.verts_vertex_lat().ndarray
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
        array_ns=xp,
    )
    assert test_helpers.dallclose(pos_on_tplane_e_x, pos_on_tplane_e_x_ref, atol=1e-6, rtol=1e-7)
    assert test_helpers.dallclose(pos_on_tplane_e_y, pos_on_tplane_e_y_ref, atol=1e-6, rtol=1e-7)
