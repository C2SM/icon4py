# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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

import logging
from unittest import mock

import gt4py.next as gtx
import pytest
from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.dycore.state_utils import states as solve_nh_states
from icon4py.model.common import constants, dimension as dims, utils as common_utils
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.grid.vertical import VerticalGridConfig
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from icon4pytools.py2fgen.wrappers import dycore_wrapper, wrapper_dimension as w_dim

from . import utils


logging.basicConfig(level=logging.INFO)


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init,jstep_init, istep_exit,jstep_exit", [(1, 0, 2, 0)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", (2,))
def test_dycore_wrapper_granule_inputs(
    istep_init,
    istep_exit,
    jstep_init,
    jstep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
    caplog,
    icon_grid,
):
    caplog.set_level(logging.DEBUG)

    # savepoints
    sp = savepoint_nonhydro_init

    # --- Granule input parameters for dycore init

    # non hydrostatic config parameters
    itime_scheme = solve_nh.TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = solve_nh.RhoThetaAdvectionType.MIURA
    igradp_method = solve_nh.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    ndyn_substeps = ndyn_substeps
    rayleigh_type = constants.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = solve_nh.DivergenceDampingOrder.COMBINED
    is_iau_active = False
    iau_wgt_dyn = 1.0
    divdamp_type = 3
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    l_vert_nested = False
    rhotheta_offctr = -0.1
    veladv_offctr = 0.25
    max_nudging_coeff = 0.075
    divdamp_fac = 0.004
    divdamp_fac2 = 0.004
    divdamp_fac3 = 0.004
    divdamp_fac4 = 0.004
    divdamp_z = 32500.0
    divdamp_z2 = 40000.0
    divdamp_z3 = 60000.0
    divdamp_z4 = 80000.0

    # vertical grid params
    num_levels = 65
    lowest_layer_thickness = 20.0
    model_top_height = 23000.0
    stretch_factor = 0.65
    rayleigh_damping_height = 12500.0

    # vertical params
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    nflat_gradp = grid_savepoint.nflat_gradp()

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")

    # Cell geometry
    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    cell_areas = grid_savepoint.cell_areas()

    # Edge geometry
    tangent_orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inverse_dual_edge_lengths = grid_savepoint.inv_dual_edge_length()
    inverse_vertex_vertex_lengths = grid_savepoint.inv_vert_vert_length()
    primal_normal_vert_x = grid_savepoint.primal_normal_vert_x()
    primal_normal_vert_y = grid_savepoint.primal_normal_vert_y()
    dual_normal_vert_x = grid_savepoint.dual_normal_vert_x()
    dual_normal_vert_y = grid_savepoint.dual_normal_vert_y()
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_x = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_y = grid_savepoint.dual_normal_cell_y()
    edge_areas = grid_savepoint.edge_areas()
    f_e = grid_savepoint.f_e()
    edge_center_lat = grid_savepoint.edge_center_lat()
    edge_center_lon = grid_savepoint.edge_center_lon()
    primal_normal_x = grid_savepoint.primal_normal_x()
    primal_normal_y = grid_savepoint.primal_normal_y()

    # metric state parameters
    bdy_halo_c = metrics_savepoint.bdy_halo_c()
    mask_prog_halo_c = metrics_savepoint.mask_prog_halo_c()
    rayleigh_w = metrics_savepoint.rayleigh_w()
    exner_exfac = metrics_savepoint.exner_exfac()
    exner_ref_mc = metrics_savepoint.exner_ref_mc()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    wgtfacq_c = metrics_savepoint.wgtfacq_c_dsl()
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    rho_ref_mc = metrics_savepoint.rho_ref_mc()
    theta_ref_mc = metrics_savepoint.theta_ref_mc()
    vwind_expl_wgt = metrics_savepoint.vwind_expl_wgt()
    d_exner_dz_ref_ic = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    theta_ref_ic = metrics_savepoint.theta_ref_ic()
    d2dexdz2_fac1_mc = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc = metrics_savepoint.d2dexdz2_fac2_mc()
    rho_ref_me = metrics_savepoint.rho_ref_me()
    theta_ref_me = metrics_savepoint.theta_ref_me()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    zdiff_gradp = metrics_savepoint._get_field(
        "zdiff_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim
    )
    vertoffset_gradp = metrics_savepoint._get_field(
        "vertoffset_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
    )
    ipeidx_dsl = metrics_savepoint.ipeidx_dsl()
    pg_exdist = metrics_savepoint.pg_exdist()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    wgtfacq_e = metrics_savepoint.wgtfacq_e_dsl(num_levels)
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()
    hmask_dd3d = metrics_savepoint.hmask_dd3d()
    scalfac_dd3d = metrics_savepoint.scalfac_dd3d()
    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    coeff_gradekin = metrics_savepoint._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)

    # interpolation state parameters
    c_lin_e = interpolation_savepoint.c_lin_e()
    c_intp = interpolation_savepoint.c_intp()
    e_flx_avg = interpolation_savepoint.e_flx_avg()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    geofac_rot = interpolation_savepoint.geofac_rot()
    pos_on_tplane_e_1 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_2 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim
    )
    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    rbf_coeff_1 = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_coeff_2 = interpolation_savepoint.rbf_vec_coeff_v2()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_n2s = interpolation_savepoint.geofac_n2s()
    geofac_grg_x = interpolation_savepoint.geofac_grg()[0]
    geofac_grg_y = interpolation_savepoint.geofac_grg()[1]
    nudgecoeff_e = interpolation_savepoint.nudgecoeff_e()

    # other params
    c_owner_mask = grid_savepoint.c_owner_mask()

    # --- Set Up Grid Parameters ---
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    # raw serialised data which is not yet offset
    cell_starts = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_start_index"))
    cell_ends = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_end_index"))
    vertex_starts = gtx.as_field(
        (w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_start_index")
    )
    vertex_ends = gtx.as_field((w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_end_index"))
    edge_starts = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_start_index"))
    edge_ends = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_end_index"))

    c2e = gtx.as_field((dims.CellDim, dims.C2EDim), grid_savepoint._read_int32("c2e"))
    e2c = gtx.as_field((dims.EdgeDim, dims.E2CDim), grid_savepoint._read_int32("e2c"))
    c2e2c = gtx.as_field((dims.CellDim, dims.C2E2CDim), grid_savepoint._read_int32("c2e2c"))
    e2c2e = gtx.as_field((dims.EdgeDim, dims.E2C2EDim), grid_savepoint._read_int32("e2c2e"))
    e2v = gtx.as_field((dims.EdgeDim, dims.E2VDim), grid_savepoint._read_int32("e2v"))
    v2e = gtx.as_field((dims.VertexDim, dims.V2EDim), grid_savepoint._read_int32("v2e"))
    v2c = gtx.as_field((dims.VertexDim, dims.V2CDim), grid_savepoint._read_int32("v2c"))
    e2c2v = gtx.as_field((dims.EdgeDim, dims.E2C2VDim), grid_savepoint._read_int32("e2c2v"))
    c2v = gtx.as_field((dims.CellDim, dims.C2VDim), grid_savepoint._read_int32("c2v"))

    # global grid params
    global_root = 4
    global_level = 9

    # --- Granule input parameters for dycore run
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")
    initial_divdamp_fac = sp.divdamp_fac_o2()

    # PrepAdvection
    vn_traj = sp.vn_traj()
    mass_flx_me = sp.mass_flx_me()
    mass_flx_ic = sp.mass_flx_ic()

    # Diagnostic state parameters
    theta_v_ic = sp.theta_v_ic()
    exner_pr = sp.exner_pr()
    rho_ic = sp.rho_ic()
    ddt_exner_phy = sp.ddt_exner_phy()
    grf_tend_rho = sp.grf_tend_rho()
    grf_tend_thv = sp.grf_tend_thv()
    grf_tend_w = sp.grf_tend_w()
    mass_fl_e = sp.mass_fl_e()
    ddt_vn_phy = sp.ddt_vn_phy()
    grf_tend_vn = sp.grf_tend_vn()
    ddt_vn_apc_ntl1 = sp.ddt_vn_apc_pc(1)
    ddt_vn_apc_ntl2 = sp.ddt_vn_apc_pc(2)
    ddt_w_adv_ntl1 = sp.ddt_w_adv_pc(1)
    ddt_w_adv_ntl2 = sp.ddt_w_adv_pc(2)
    vt = sp.vt()
    vn_ie = sp.vn_ie()
    w_concorr_c = sp.w_concorr_c()
    exner_dyn_incr = sp.exner_dyn_incr()

    # Prognostic state parameters
    w_now = sp.w_now()
    vn_now = sp.vn_now()
    theta_v_now = sp.theta_v_now()
    rho_now = sp.rho_now()
    exner_now = sp.exner_now()

    w_new = sp.w_new()
    vn_new = sp.vn_new()
    theta_v_new = sp.theta_v_new()
    rho_new = sp.rho_new()
    exner_new = sp.exner_new()

    # using fortran indices
    nnow = 1
    nnew = 2
    jstep_init_fortran = jstep_init + 1

    # --- Expected objects that form inputs into init function ---
    expected_icon_grid = icon_grid
    expected_edge_geometry = grid_savepoint.construct_edge_geometry()
    expected_cell_geometry = grid_savepoint.construct_cell_geometry()
    expected_interpolation_state = solve_nh_states.InterpolationState(
        c_lin_e=interpolation_savepoint.c_lin_e(),
        c_intp=interpolation_savepoint.c_intp(),
        e_flx_avg=interpolation_savepoint.e_flx_avg(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        geofac_rot=interpolation_savepoint.geofac_rot(),
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        rbf_vec_coeff_e=interpolation_savepoint.rbf_vec_coeff_e(),
        e_bln_c_s=helpers.as_1D_sparse_field(interpolation_savepoint.e_bln_c_s(), dims.CEDim),
        rbf_coeff_1=interpolation_savepoint.rbf_vec_coeff_v1(),
        rbf_coeff_2=interpolation_savepoint.rbf_vec_coeff_v2(),
        geofac_div=helpers.as_1D_sparse_field(interpolation_savepoint.geofac_div(), dims.CEDim),
        geofac_n2s=interpolation_savepoint.geofac_n2s(),
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
    )
    expected_metric_state = solve_nh_states.MetricStateNonHydro(
        bdy_halo_c=metrics_savepoint.bdy_halo_c(),
        mask_prog_halo_c=metrics_savepoint.mask_prog_halo_c(),
        rayleigh_w=metrics_savepoint.rayleigh_w(),
        exner_exfac=metrics_savepoint.exner_exfac(),
        exner_ref_mc=metrics_savepoint.exner_ref_mc(),
        wgtfac_c=metrics_savepoint.wgtfac_c(),
        wgtfacq_c=metrics_savepoint.wgtfacq_c_dsl(),
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        rho_ref_mc=metrics_savepoint.rho_ref_mc(),
        theta_ref_mc=metrics_savepoint.theta_ref_mc(),
        vwind_expl_wgt=metrics_savepoint.vwind_expl_wgt(),
        d_exner_dz_ref_ic=metrics_savepoint.d_exner_dz_ref_ic(),
        ddqz_z_half=metrics_savepoint.ddqz_z_half(),
        theta_ref_ic=metrics_savepoint.theta_ref_ic(),
        d2dexdz2_fac1_mc=metrics_savepoint.d2dexdz2_fac1_mc(),
        d2dexdz2_fac2_mc=metrics_savepoint.d2dexdz2_fac2_mc(),
        rho_ref_me=metrics_savepoint.rho_ref_me(),
        theta_ref_me=metrics_savepoint.theta_ref_me(),
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        vertoffset_gradp=metrics_savepoint.vertoffset_gradp(),
        ipeidx_dsl=metrics_savepoint.ipeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        ddqz_z_full_e=metrics_savepoint.ddqz_z_full_e(),
        ddxt_z_full=metrics_savepoint.ddxt_z_full(),
        wgtfac_e=metrics_savepoint.wgtfac_e(),
        wgtfacq_e=metrics_savepoint.wgtfacq_e_dsl(num_levels),
        vwind_impl_wgt=metrics_savepoint.vwind_impl_wgt(),
        hmask_dd3d=metrics_savepoint.hmask_dd3d(),
        scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
        coeff1_dwdz=metrics_savepoint.coeff1_dwdz(),
        coeff2_dwdz=metrics_savepoint.coeff2_dwdz(),
        coeff_gradekin=metrics_savepoint.coeff_gradekin(),
    )
    expected_vertical_config = VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    expected_vertical_params = v_grid.VerticalGrid(
        config=expected_vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    expected_config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    expected_additional_parameters = solve_nh.NonHydrostaticParams(expected_config)

    # --- Expected objects that form inputs into run function ---
    expected_diagnostic_state_nh = solve_nh_states.DiagnosticStateNonHydro(
        theta_v_ic=sp.theta_v_ic(),
        exner_pr=sp.exner_pr(),
        rho_ic=sp.rho_ic(),
        ddt_exner_phy=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_fl_e=sp.mass_fl_e(),
        ddt_vn_phy=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        ddt_vn_apc_pc=common_utils.Pair(sp.ddt_vn_apc_pc(1), sp.ddt_vn_apc_pc(2)),
        ddt_w_adv_pc=common_utils.Pair(sp.ddt_w_adv_pc(1), ddt_w_adv_ntl2=sp.ddt_w_adv_pc(2)),
        vt=sp.vt(),
        vn_ie=sp.vn_ie(),
        w_concorr_c=sp.w_concorr_c(),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=sp.exner_dyn_incr(),
    )
    prognostic_state_nnow = prognostics.PrognosticState(
        w=sp.w_now(),
        vn=sp.vn_now(),
        theta_v=sp.theta_v_now(),
        rho=sp.rho_now(),
        exner=sp.exner_now(),
    )
    prognostic_state_nnew = prognostics.PrognosticState(
        w=sp.w_new(),
        vn=sp.vn_new(),
        theta_v=sp.theta_v_new(),
        rho=sp.rho_new(),
        exner=sp.exner_new(),
    )
    expected_prognostic_state_swp = common_utils.NextStepPair(
        prognostic_state_nnow, prognostic_state_nnew
    )

    expected_prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
    )
    expected_initial_divdamp_fac = sp.divdamp_fac_o2()
    expected_dtime = sp.get_metadata("dtime").get("dtime")
    expected_recompute = sp.get_metadata("recompute").get("recompute")
    expected_linit = sp.get_metadata("linit").get("linit")
    expected_clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    expected_lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    expected_nnow = 0
    expected_nnew = 1
    expected_at_first_substep = jstep_init == 0
    expected_at_last_substep = jstep_init == (ndyn_substeps - 1)

    # --- Initialize the Grid ---
    dycore_wrapper.grid_init(
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    # --- Mock and Test SolveNonhydro.init ---
    with mock.patch(
        "icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro.SolveNonhydro.__init__",
        return_value=None,
    ) as mock_init:
        dycore_wrapper.solve_nh_init(
            vct_a=vct_a,
            vct_b=vct_b,
            cell_areas=cell_areas,
            primal_normal_cell_x=primal_normal_cell_x,
            primal_normal_cell_y=primal_normal_cell_y,
            dual_normal_cell_x=dual_normal_cell_x,
            dual_normal_cell_y=dual_normal_cell_y,
            edge_areas=edge_areas,
            tangent_orientation=tangent_orientation,
            inverse_primal_edge_lengths=inverse_primal_edge_lengths,
            inverse_dual_edge_lengths=inverse_dual_edge_lengths,
            inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
            primal_normal_vert_x=primal_normal_vert_x,
            primal_normal_vert_y=primal_normal_vert_y,
            dual_normal_vert_x=dual_normal_vert_x,
            dual_normal_vert_y=dual_normal_vert_y,
            f_e=f_e,
            c_lin_e=c_lin_e,
            c_intp=c_intp,
            e_flx_avg=e_flx_avg,
            geofac_grdiv=geofac_grdiv,
            geofac_rot=geofac_rot,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            rbf_vec_coeff_e=rbf_vec_coeff_e,
            e_bln_c_s=e_bln_c_s,
            rbf_coeff_1=rbf_coeff_1,
            rbf_coeff_2=rbf_coeff_2,
            geofac_div=geofac_div,
            geofac_n2s=geofac_n2s,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            nudgecoeff_e=nudgecoeff_e,
            bdy_halo_c=bdy_halo_c,
            mask_prog_halo_c=mask_prog_halo_c,
            rayleigh_w=rayleigh_w,
            exner_exfac=exner_exfac,
            exner_ref_mc=exner_ref_mc,
            wgtfac_c=wgtfac_c,
            wgtfacq_c=wgtfacq_c,
            inv_ddqz_z_full=inv_ddqz_z_full,
            rho_ref_mc=rho_ref_mc,
            theta_ref_mc=theta_ref_mc,
            vwind_expl_wgt=vwind_expl_wgt,
            d_exner_dz_ref_ic=d_exner_dz_ref_ic,
            ddqz_z_half=ddqz_z_half,
            theta_ref_ic=theta_ref_ic,
            d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
            d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
            rho_ref_me=rho_ref_me,
            theta_ref_me=theta_ref_me,
            ddxn_z_full=ddxn_z_full,
            zdiff_gradp=zdiff_gradp,
            vertoffset_gradp=vertoffset_gradp,
            ipeidx_dsl=ipeidx_dsl,
            pg_exdist=pg_exdist,
            ddqz_z_full_e=ddqz_z_full_e,
            ddxt_z_full=ddxt_z_full,
            wgtfac_e=wgtfac_e,
            wgtfacq_e=wgtfacq_e,
            vwind_impl_wgt=vwind_impl_wgt,
            hmask_dd3d=hmask_dd3d,
            scalfac_dd3d=scalfac_dd3d,
            coeff1_dwdz=coeff1_dwdz,
            coeff2_dwdz=coeff2_dwdz,
            coeff_gradekin=coeff_gradekin,
            c_owner_mask=c_owner_mask,
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            edge_center_lat=edge_center_lat,
            edge_center_lon=edge_center_lon,
            primal_normal_x=primal_normal_x,
            primal_normal_y=primal_normal_y,
            rayleigh_damping_height=rayleigh_damping_height,
            itime_scheme=itime_scheme,
            iadv_rhotheta=iadv_rhotheta,
            igradp_method=igradp_method,
            ndyn_substeps=ndyn_substeps,
            rayleigh_type=rayleigh_type,
            rayleigh_coeff=rayleigh_coeff,
            divdamp_order=divdamp_order,
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
            divdamp_type=divdamp_type,
            divdamp_trans_start=divdamp_trans_start,
            divdamp_trans_end=divdamp_trans_end,
            l_vert_nested=l_vert_nested,
            rhotheta_offctr=rhotheta_offctr,
            veladv_offctr=veladv_offctr,
            max_nudging_coeff=max_nudging_coeff,
            divdamp_fac=divdamp_fac,
            divdamp_fac2=divdamp_fac2,
            divdamp_fac3=divdamp_fac3,
            divdamp_fac4=divdamp_fac4,
            divdamp_z=divdamp_z,
            divdamp_z2=divdamp_z2,
            divdamp_z3=divdamp_z3,
            divdamp_z4=divdamp_z4,
            lowest_layer_thickness=lowest_layer_thickness,
            model_top_height=model_top_height,
            stretch_factor=stretch_factor,
            nflat_gradp=nflat_gradp,
            num_levels=num_levels,
        )

        # Check input arguments to SolveNonhydro.init
        captured_args, captured_kwargs = mock_init.call_args

        # special case of grid._id as we do not use this arg in the wrapper as we cant pass strings from Fortran to the wrapper
        try:
            result, error_message = utils.compare_objects(
                captured_kwargs["grid"], expected_icon_grid
            )
            assert result, f"Grid comparison failed: {error_message}"
        except AssertionError as e:
            error_message = str(e)
            if "object.connectivities" not in error_message:
                raise
            else:
                pass

        result, error_message = utils.compare_objects(captured_kwargs["config"], expected_config)
        assert result, f"Config comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["params"], expected_additional_parameters
        )
        assert result, f"Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["metric_state_nonhydro"], expected_metric_state
        )
        assert result, f"Metric State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["interpolation_state"], expected_interpolation_state
        )
        assert result, f"Interpolation State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["vertical_params"], expected_vertical_params
        )
        assert result, f"Vertical Params comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["edge_geometry"], expected_edge_geometry
        )
        assert result, f"Edge Geometry comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["cell_geometry"], expected_cell_geometry
        )
        assert result, f"Cell Geometry comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["owner_mask"], grid_savepoint.c_owner_mask()
        )
        assert result, f"Owner Mask comparison failed: {error_message}"

    # --- Mock and Test SolveNonhydro.run ---
    with mock.patch(
        "icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro.SolveNonhydro.time_step"
    ) as mock_init:
        dycore_wrapper.solve_nh_run(
            rho_now=rho_now,
            rho_new=rho_new,
            exner_now=exner_now,
            exner_new=exner_new,
            w_now=w_now,
            w_new=w_new,
            theta_v_now=theta_v_now,
            theta_v_new=theta_v_new,
            vn_now=vn_now,
            vn_new=vn_new,
            w_concorr_c=w_concorr_c,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            exner_pr=exner_pr,
            exner_dyn_incr=exner_dyn_incr,
            ddt_exner_phy=ddt_exner_phy,
            grf_tend_rho=grf_tend_rho,
            grf_tend_thv=grf_tend_thv,
            grf_tend_w=grf_tend_w,
            mass_fl_e=mass_fl_e,
            ddt_vn_phy=ddt_vn_phy,
            grf_tend_vn=grf_tend_vn,
            vn_ie=vn_ie,
            vt=vt,
            mass_flx_me=mass_flx_me,
            mass_flx_ic=mass_flx_ic,
            vn_traj=vn_traj,
            dtime=dtime,
            lprep_adv=lprep_adv,
            clean_mflx=clean_mflx,
            recompute=recompute,
            linit=linit,
            divdamp_fac_o2=initial_divdamp_fac,
            ndyn_substeps=ndyn_substeps,
            idyn_timestep=jstep_init_fortran,
            nnow=nnow,
            nnew=nnew,
        )

        # Check input arguments to SolveNonhydro.time_step
        captured_args, captured_kwargs = mock_init.call_args

        result, error_message = utils.compare_objects(
            captured_kwargs["diagnostic_state_nh"], expected_diagnostic_state_nh
        )
        assert result, f"Diagnostic State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["prognostic_state_swp"], expected_prognostic_state_swp
        )
        assert result, f"Prognostic State comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["prep_adv"], expected_prep_adv
        )
        assert result, f"Prep Advection comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["divdamp_fac_o2"], expected_initial_divdamp_fac
        )
        assert result, f"Divdamp Factor comparison failed: {error_message}"

        result, error_message = utils.compare_objects(captured_kwargs["dtime"], expected_dtime)
        assert result, f"dtime comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["l_recompute"], expected_recompute
        )
        assert result, f"Recompute flag comparison failed: {error_message}"

        result, error_message = utils.compare_objects(captured_kwargs["l_init"], expected_linit)
        assert result, f"Init flag comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["lclean_mflx"], expected_clean_mflx
        )
        assert result, f"Clean MFLX flag comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["lprep_adv"], expected_lprep_adv
        )
        assert result, f"Prep Advection flag comparison failed: {error_message}"

        result, error_message = utils.compare_objects(captured_kwargs["nnew"], expected_nnew)
        assert result, f"nnew comparison failed: {error_message}"

        result, error_message = utils.compare_objects(captured_kwargs["nnow"], expected_nnow)
        assert result, f"nnow comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["at_first_substep"], expected_at_first_substep
        )
        assert result, f"First Substep comparison failed: {error_message}"

        result, error_message = utils.compare_objects(
            captured_kwargs["at_last_substep"], expected_at_last_substep
        )
        assert result, f"Last Substep comparison failed: {error_message}"


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init,jstep_init, istep_exit,jstep_exit", [(1, 0, 2, 0)])
@pytest.mark.parametrize(
    "experiment,step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
    ],
)
def test_granule_solve_nonhydro_single_step_regional(
    istep_init,
    istep_exit,
    jstep_init,
    jstep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
    caplog,
    icon_grid,
):
    caplog.set_level(logging.DEBUG)

    # savepoints
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit

    # non hydrostatic config parameters
    itime_scheme = solve_nh.TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = solve_nh.RhoThetaAdvectionType.MIURA
    igradp_method = solve_nh.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    ndyn_substeps = ndyn_substeps
    rayleigh_type = constants.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = solve_nh.DivergenceDampingOrder.COMBINED
    is_iau_active = False
    iau_wgt_dyn = 1.0
    divdamp_type = 3
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    l_vert_nested = False
    rhotheta_offctr = -0.1
    veladv_offctr = 0.25
    max_nudging_coeff = 0.075
    divdamp_fac = 0.004
    divdamp_fac2 = 0.004
    divdamp_fac3 = 0.004
    divdamp_fac4 = 0.004
    divdamp_z = 32500.0
    divdamp_z2 = 40000.0
    divdamp_z3 = 60000.0
    divdamp_z4 = 80000.0

    # vertical grid params
    num_levels = 65
    lowest_layer_thickness = 20.0
    model_top_height = 23000.0
    stretch_factor = 0.65
    rayleigh_damping_height = 12500.0

    # vertical params
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    nflat_gradp = grid_savepoint.nflat_gradp()

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")

    # Cell geometry
    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    cell_areas = grid_savepoint.cell_areas()

    # Edge geometry
    tangent_orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inverse_dual_edge_lengths = grid_savepoint.inv_dual_edge_length()
    inverse_vertex_vertex_lengths = grid_savepoint.inv_vert_vert_length()
    primal_normal_vert_x = grid_savepoint.primal_normal_vert_x()
    primal_normal_vert_y = grid_savepoint.primal_normal_vert_y()
    dual_normal_vert_x = grid_savepoint.dual_normal_vert_x()
    dual_normal_vert_y = grid_savepoint.dual_normal_vert_y()
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_x = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_y = grid_savepoint.dual_normal_cell_y()
    edge_areas = grid_savepoint.edge_areas()
    f_e = grid_savepoint.f_e()
    edge_center_lat = grid_savepoint.edge_center_lat()
    edge_center_lon = grid_savepoint.edge_center_lon()
    primal_normal_x = grid_savepoint.primal_normal_x()
    primal_normal_y = grid_savepoint.primal_normal_y()

    # metric state parameters
    bdy_halo_c = metrics_savepoint.bdy_halo_c()
    mask_prog_halo_c = metrics_savepoint.mask_prog_halo_c()
    rayleigh_w = metrics_savepoint.rayleigh_w()
    exner_exfac = metrics_savepoint.exner_exfac()
    exner_ref_mc = metrics_savepoint.exner_ref_mc()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    wgtfacq_c = metrics_savepoint.wgtfacq_c_dsl()
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    rho_ref_mc = metrics_savepoint.rho_ref_mc()
    theta_ref_mc = metrics_savepoint.theta_ref_mc()
    vwind_expl_wgt = metrics_savepoint.vwind_expl_wgt()
    d_exner_dz_ref_ic = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    theta_ref_ic = metrics_savepoint.theta_ref_ic()
    d2dexdz2_fac1_mc = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc = metrics_savepoint.d2dexdz2_fac2_mc()
    rho_ref_me = metrics_savepoint.rho_ref_me()
    theta_ref_me = metrics_savepoint.theta_ref_me()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    zdiff_gradp = metrics_savepoint._get_field(
        "zdiff_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim
    )
    vertoffset_gradp = metrics_savepoint._get_field(
        "vertoffset_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
    )
    ipeidx_dsl = metrics_savepoint.ipeidx_dsl()
    pg_exdist = metrics_savepoint.pg_exdist()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    wgtfacq_e = metrics_savepoint.wgtfacq_e_dsl(num_levels)
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()
    hmask_dd3d = metrics_savepoint.hmask_dd3d()
    scalfac_dd3d = metrics_savepoint.scalfac_dd3d()
    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    coeff_gradekin = metrics_savepoint._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)

    # interpolation state parameters
    c_lin_e = interpolation_savepoint.c_lin_e()
    c_intp = interpolation_savepoint.c_intp()
    e_flx_avg = interpolation_savepoint.e_flx_avg()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    geofac_rot = interpolation_savepoint.geofac_rot()
    pos_on_tplane_e_1 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_2 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim
    )
    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    rbf_coeff_1 = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_coeff_2 = interpolation_savepoint.rbf_vec_coeff_v2()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_n2s = interpolation_savepoint.geofac_n2s()
    geofac_grg_x = interpolation_savepoint.geofac_grg()[0]
    geofac_grg_y = interpolation_savepoint.geofac_grg()[1]
    nudgecoeff_e = interpolation_savepoint.nudgecoeff_e()

    # other params
    c_owner_mask = grid_savepoint.c_owner_mask()

    # --- Set Up Grid Parameters ---
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    # raw serialised data which is not yet offset
    cell_starts = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_start_index"))
    cell_ends = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_end_index"))
    vertex_starts = gtx.as_field(
        (w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_start_index")
    )
    vertex_ends = gtx.as_field((w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_end_index"))
    edge_starts = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_start_index"))
    edge_ends = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_end_index"))

    c2e = gtx.as_field((dims.CellDim, dims.C2EDim), grid_savepoint._read_int32("c2e"))
    e2c = gtx.as_field((dims.EdgeDim, dims.E2CDim), grid_savepoint._read_int32("e2c"))
    c2e2c = gtx.as_field((dims.CellDim, dims.C2E2CDim), grid_savepoint._read_int32("c2e2c"))
    e2c2e = gtx.as_field((dims.EdgeDim, dims.E2C2EDim), grid_savepoint._read_int32("e2c2e"))
    e2v = gtx.as_field((dims.EdgeDim, dims.E2VDim), grid_savepoint._read_int32("e2v"))
    v2e = gtx.as_field((dims.VertexDim, dims.V2EDim), grid_savepoint._read_int32("v2e"))
    v2c = gtx.as_field((dims.VertexDim, dims.V2CDim), grid_savepoint._read_int32("v2c"))
    e2c2v = gtx.as_field((dims.EdgeDim, dims.E2C2VDim), grid_savepoint._read_int32("e2c2v"))
    c2v = gtx.as_field((dims.CellDim, dims.C2VDim), grid_savepoint._read_int32("c2v"))

    # global grid params
    global_root = 4
    global_level = 9

    dycore_wrapper.grid_init(
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    # call solve init
    dycore_wrapper.solve_nh_init(
        vct_a=vct_a,
        vct_b=vct_b,
        cell_areas=cell_areas,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_areas=edge_areas,
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inverse_dual_edge_lengths,
        inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        f_e=f_e,
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
        bdy_halo_c=bdy_halo_c,
        mask_prog_halo_c=mask_prog_halo_c,
        rayleigh_w=rayleigh_w,
        exner_exfac=exner_exfac,
        exner_ref_mc=exner_ref_mc,
        wgtfac_c=wgtfac_c,
        wgtfacq_c=wgtfacq_c,
        inv_ddqz_z_full=inv_ddqz_z_full,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        vwind_expl_wgt=vwind_expl_wgt,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        ddxn_z_full=ddxn_z_full,
        zdiff_gradp=zdiff_gradp,
        vertoffset_gradp=vertoffset_gradp,
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=coeff_gradekin,
        c_owner_mask=c_owner_mask,
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
        rayleigh_damping_height=rayleigh_damping_height,
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        ndyn_substeps=ndyn_substeps,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        divdamp_order=divdamp_order,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        divdamp_type=divdamp_type,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        l_vert_nested=l_vert_nested,
        rhotheta_offctr=rhotheta_offctr,
        veladv_offctr=veladv_offctr,
        max_nudging_coeff=max_nudging_coeff,
        divdamp_fac=divdamp_fac,
        divdamp_fac2=divdamp_fac2,
        divdamp_fac3=divdamp_fac3,
        divdamp_fac4=divdamp_fac4,
        divdamp_z=divdamp_z,
        divdamp_z2=divdamp_z2,
        divdamp_z3=divdamp_z3,
        divdamp_z4=divdamp_z4,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        nflat_gradp=nflat_gradp,
        num_levels=num_levels,
    )

    # solve nh run parameters
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")
    initial_divdamp_fac = sp.divdamp_fac_o2()

    # PrepAdvection
    vn_traj = sp.vn_traj()
    mass_flx_me = sp.mass_flx_me()
    mass_flx_ic = sp.mass_flx_ic()

    # Diagnostic state parameters
    theta_v_ic = sp.theta_v_ic()
    exner_pr = sp.exner_pr()
    rho_ic = sp.rho_ic()
    ddt_exner_phy = sp.ddt_exner_phy()
    grf_tend_rho = sp.grf_tend_rho()
    grf_tend_thv = sp.grf_tend_thv()
    grf_tend_w = sp.grf_tend_w()
    mass_fl_e = sp.mass_fl_e()
    ddt_vn_phy = sp.ddt_vn_phy()
    grf_tend_vn = sp.grf_tend_vn()
    ddt_vn_apc_ntl1 = sp.ddt_vn_apc_pc(1)
    ddt_vn_apc_ntl2 = sp.ddt_vn_apc_pc(2)
    ddt_w_adv_ntl1 = sp.ddt_w_adv_pc(1)
    ddt_w_adv_ntl2 = sp.ddt_w_adv_pc(2)
    vt = sp.vt()
    vn_ie = sp.vn_ie()
    w_concorr_c = sp.w_concorr_c()
    exner_dyn_incr = sp.exner_dyn_incr()

    # Prognostic state parameters
    w_now = sp.w_now()
    vn_now = sp.vn_now()
    theta_v_now = sp.theta_v_now()
    rho_now = sp.rho_now()
    exner_now = sp.exner_now()

    w_new = sp.w_new()
    vn_new = sp.vn_new()
    theta_v_new = sp.theta_v_new()
    rho_new = sp.rho_new()
    exner_new = sp.exner_new()

    # using fortran indices
    nnow = 1
    nnew = 2
    jstep_init_fortran = jstep_init + 1

    dycore_wrapper.solve_nh_run(
        rho_now=rho_now,
        rho_new=rho_new,
        exner_now=exner_now,
        exner_new=exner_new,
        w_now=w_now,
        w_new=w_new,
        theta_v_now=theta_v_now,
        theta_v_new=theta_v_new,
        vn_now=vn_now,
        vn_new=vn_new,
        w_concorr_c=w_concorr_c,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
        ddt_w_adv_ntl1=ddt_w_adv_ntl1,
        ddt_w_adv_ntl2=ddt_w_adv_ntl2,
        theta_v_ic=theta_v_ic,
        rho_ic=rho_ic,
        exner_pr=exner_pr,
        exner_dyn_incr=exner_dyn_incr,
        ddt_exner_phy=ddt_exner_phy,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_fl_e=mass_fl_e,
        ddt_vn_phy=ddt_vn_phy,
        grf_tend_vn=grf_tend_vn,
        vn_ie=vn_ie,
        vt=vt,
        mass_flx_me=mass_flx_me,
        mass_flx_ic=mass_flx_ic,
        vn_traj=vn_traj,
        dtime=dtime,
        lprep_adv=lprep_adv,
        clean_mflx=clean_mflx,
        recompute=recompute,
        linit=linit,
        divdamp_fac_o2=initial_divdamp_fac,
        ndyn_substeps=ndyn_substeps,
        idyn_timestep=jstep_init_fortran,
        nnow=nnow,
        nnew=nnew,
    )

    assert helpers.dallclose(
        theta_v_new.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(exner_new.asnumpy(), sp_step_exit.exner_new().asnumpy())

    assert helpers.dallclose(
        vn_new.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-12,
        atol=1e-13,
    )

    assert helpers.dallclose(rho_new.asnumpy(), savepoint_nonhydro_exit.rho_new().asnumpy())

    assert helpers.dallclose(
        w_new.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        exner_dyn_incr.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


@pytest.mark.slow_tests
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize(
    "istep_init, jstep_init, step_date_init, istep_exit, jstep_exit, step_date_exit, vn_only",
    [
        (1, 0, "2021-06-20T12:00:10.000", 2, 1, "2021-06-20T12:00:10.000", False),
        (1, 0, "2021-06-20T12:00:20.000", 2, 1, "2021-06-20T12:00:20.000", True),
    ],
)
def test_granule_solve_nonhydro_multi_step_regional(
    step_date_init,
    step_date_exit,
    istep_exit,
    jstep_init,
    jstep_exit,
    icon_grid,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    vn_only,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_exit,
    experiment,
    ndyn_substeps,
):
    # savepoints
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit

    # non hydrostatic config parameters
    itime_scheme = solve_nh.TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = solve_nh.RhoThetaAdvectionType.MIURA
    igradp_method = solve_nh.HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    ndyn_substeps = ndyn_substeps
    rayleigh_type = constants.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = solve_nh.DivergenceDampingOrder.COMBINED
    is_iau_active = False
    iau_wgt_dyn = 1.0
    divdamp_type = 3
    divdamp_trans_start = 12500.0
    divdamp_trans_end = 17500.0
    l_vert_nested = False
    rhotheta_offctr = -0.1
    veladv_offctr = 0.25
    max_nudging_coeff = 0.075
    divdamp_fac = 0.004
    divdamp_fac2 = 0.004
    divdamp_fac3 = 0.004
    divdamp_fac4 = 0.004
    divdamp_z = 32500.0
    divdamp_z2 = 40000.0
    divdamp_z3 = 60000.0
    divdamp_z4 = 80000.0

    # vertical grid params
    num_levels = 65
    lowest_layer_thickness = 20.0
    model_top_height = 23000.0
    stretch_factor = 0.65
    rayleigh_damping_height = 12500.0

    # vertical params
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    nflat_gradp = grid_savepoint.nflat_gradp()

    # other params
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")

    # Cell geometry
    cell_center_lat = grid_savepoint.cell_center_lat()
    cell_center_lon = grid_savepoint.cell_center_lon()
    cell_areas = grid_savepoint.cell_areas()

    # Edge geometry
    tangent_orientation = grid_savepoint.tangent_orientation()
    inverse_primal_edge_lengths = grid_savepoint.inverse_primal_edge_lengths()
    inverse_dual_edge_lengths = grid_savepoint.inv_dual_edge_length()
    inverse_vertex_vertex_lengths = grid_savepoint.inv_vert_vert_length()
    primal_normal_vert_x = grid_savepoint.primal_normal_vert_x()
    primal_normal_vert_y = grid_savepoint.primal_normal_vert_y()
    dual_normal_vert_x = grid_savepoint.dual_normal_vert_x()
    dual_normal_vert_y = grid_savepoint.dual_normal_vert_y()
    primal_normal_cell_x = grid_savepoint.primal_normal_cell_x()
    primal_normal_cell_y = grid_savepoint.primal_normal_cell_y()
    dual_normal_cell_x = grid_savepoint.dual_normal_cell_x()
    dual_normal_cell_y = grid_savepoint.dual_normal_cell_y()
    edge_areas = grid_savepoint.edge_areas()
    f_e = grid_savepoint.f_e()
    edge_center_lat = grid_savepoint.edge_center_lat()
    edge_center_lon = grid_savepoint.edge_center_lon()
    primal_normal_x = grid_savepoint.primal_normal_x()
    primal_normal_y = grid_savepoint.primal_normal_y()

    # metric state parameters
    bdy_halo_c = metrics_savepoint.bdy_halo_c()
    mask_prog_halo_c = metrics_savepoint.mask_prog_halo_c()
    rayleigh_w = metrics_savepoint.rayleigh_w()
    exner_exfac = metrics_savepoint.exner_exfac()
    exner_ref_mc = metrics_savepoint.exner_ref_mc()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    wgtfacq_c = metrics_savepoint.wgtfacq_c_dsl()
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()
    rho_ref_mc = metrics_savepoint.rho_ref_mc()
    theta_ref_mc = metrics_savepoint.theta_ref_mc()
    vwind_expl_wgt = metrics_savepoint.vwind_expl_wgt()
    d_exner_dz_ref_ic = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    theta_ref_ic = metrics_savepoint.theta_ref_ic()
    d2dexdz2_fac1_mc = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc = metrics_savepoint.d2dexdz2_fac2_mc()
    rho_ref_me = metrics_savepoint.rho_ref_me()
    theta_ref_me = metrics_savepoint.theta_ref_me()
    ddxn_z_full = metrics_savepoint.ddxn_z_full()
    zdiff_gradp = metrics_savepoint._get_field(
        "zdiff_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim
    )
    vertoffset_gradp = metrics_savepoint._get_field(
        "vertoffset_gradp_dsl", dims.EdgeDim, dims.E2CDim, dims.KDim, dtype=gtx.int32
    )
    ipeidx_dsl = metrics_savepoint.ipeidx_dsl()
    pg_exdist = metrics_savepoint.pg_exdist()
    ddqz_z_full_e = metrics_savepoint.ddqz_z_full_e()
    ddxt_z_full = metrics_savepoint.ddxt_z_full()
    wgtfac_e = metrics_savepoint.wgtfac_e()
    wgtfacq_e = metrics_savepoint.wgtfacq_e_dsl(num_levels)
    vwind_impl_wgt = metrics_savepoint.vwind_impl_wgt()
    hmask_dd3d = metrics_savepoint.hmask_dd3d()
    scalfac_dd3d = metrics_savepoint.scalfac_dd3d()
    coeff1_dwdz = metrics_savepoint.coeff1_dwdz()
    coeff2_dwdz = metrics_savepoint.coeff2_dwdz()
    coeff_gradekin = metrics_savepoint._get_field("coeff_gradekin", dims.EdgeDim, dims.E2CDim)

    # interpolation state parameters
    c_lin_e = interpolation_savepoint.c_lin_e()
    c_intp = interpolation_savepoint.c_intp()
    e_flx_avg = interpolation_savepoint.e_flx_avg()
    geofac_grdiv = interpolation_savepoint.geofac_grdiv()
    geofac_rot = interpolation_savepoint.geofac_rot()
    pos_on_tplane_e_1 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_x", dims.EdgeDim, dims.E2CDim
    )
    pos_on_tplane_e_2 = interpolation_savepoint._get_field(
        "pos_on_tplane_e_y", dims.EdgeDim, dims.E2CDim
    )
    rbf_vec_coeff_e = interpolation_savepoint.rbf_vec_coeff_e()
    e_bln_c_s = interpolation_savepoint.e_bln_c_s()
    rbf_coeff_1 = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_coeff_2 = interpolation_savepoint.rbf_vec_coeff_v2()
    geofac_div = interpolation_savepoint.geofac_div()
    geofac_n2s = interpolation_savepoint.geofac_n2s()
    geofac_grg_x = interpolation_savepoint.geofac_grg()[0]
    geofac_grg_y = interpolation_savepoint.geofac_grg()[1]
    nudgecoeff_e = interpolation_savepoint.nudgecoeff_e()

    # other params
    c_owner_mask = grid_savepoint.c_owner_mask()

    # --- Set Up Grid Parameters ---
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")

    # raw serialised data which is not yet offset
    cell_starts = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_start_index"))
    cell_ends = gtx.as_field((w_dim.CellIndexDim,), grid_savepoint._read_int32("c_end_index"))
    vertex_starts = gtx.as_field(
        (w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_start_index")
    )
    vertex_ends = gtx.as_field((w_dim.VertexIndexDim,), grid_savepoint._read_int32("v_end_index"))
    edge_starts = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_start_index"))
    edge_ends = gtx.as_field((w_dim.EdgeIndexDim,), grid_savepoint._read_int32("e_end_index"))

    c2e = gtx.as_field((dims.CellDim, dims.C2EDim), grid_savepoint._read_int32("c2e"))
    e2c = gtx.as_field((dims.EdgeDim, dims.E2CDim), grid_savepoint._read_int32("e2c"))
    c2e2c = gtx.as_field((dims.CellDim, dims.C2E2CDim), grid_savepoint._read_int32("c2e2c"))
    e2c2e = gtx.as_field((dims.EdgeDim, dims.E2C2EDim), grid_savepoint._read_int32("e2c2e"))
    e2v = gtx.as_field((dims.EdgeDim, dims.E2VDim), grid_savepoint._read_int32("e2v"))
    v2e = gtx.as_field((dims.VertexDim, dims.V2EDim), grid_savepoint._read_int32("v2e"))
    v2c = gtx.as_field((dims.VertexDim, dims.V2CDim), grid_savepoint._read_int32("v2c"))
    e2c2v = gtx.as_field((dims.EdgeDim, dims.E2C2VDim), grid_savepoint._read_int32("e2c2v"))
    c2v = gtx.as_field((dims.CellDim, dims.C2VDim), grid_savepoint._read_int32("c2v"))

    # global grid params
    global_root = 4
    global_level = 9

    dycore_wrapper.grid_init(
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    # call solve init
    dycore_wrapper.solve_nh_init(
        vct_a=vct_a,
        vct_b=vct_b,
        cell_areas=cell_areas,
        primal_normal_cell_x=primal_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_x=dual_normal_cell_x,
        dual_normal_cell_y=dual_normal_cell_y,
        edge_areas=edge_areas,
        tangent_orientation=tangent_orientation,
        inverse_primal_edge_lengths=inverse_primal_edge_lengths,
        inverse_dual_edge_lengths=inverse_dual_edge_lengths,
        inverse_vertex_vertex_lengths=inverse_vertex_vertex_lengths,
        primal_normal_vert_x=primal_normal_vert_x,
        primal_normal_vert_y=primal_normal_vert_y,
        dual_normal_vert_x=dual_normal_vert_x,
        dual_normal_vert_y=dual_normal_vert_y,
        f_e=f_e,
        c_lin_e=c_lin_e,
        c_intp=c_intp,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        geofac_rot=geofac_rot,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        e_bln_c_s=e_bln_c_s,
        rbf_coeff_1=rbf_coeff_1,
        rbf_coeff_2=rbf_coeff_2,
        geofac_div=geofac_div,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        nudgecoeff_e=nudgecoeff_e,
        bdy_halo_c=bdy_halo_c,
        mask_prog_halo_c=mask_prog_halo_c,
        rayleigh_w=rayleigh_w,
        exner_exfac=exner_exfac,
        exner_ref_mc=exner_ref_mc,
        wgtfac_c=wgtfac_c,
        wgtfacq_c=wgtfacq_c,
        inv_ddqz_z_full=inv_ddqz_z_full,
        rho_ref_mc=rho_ref_mc,
        theta_ref_mc=theta_ref_mc,
        vwind_expl_wgt=vwind_expl_wgt,
        d_exner_dz_ref_ic=d_exner_dz_ref_ic,
        ddqz_z_half=ddqz_z_half,
        theta_ref_ic=theta_ref_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        ddxn_z_full=ddxn_z_full,
        zdiff_gradp=zdiff_gradp,
        vertoffset_gradp=vertoffset_gradp,
        ipeidx_dsl=ipeidx_dsl,
        pg_exdist=pg_exdist,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        wgtfacq_e=wgtfacq_e,
        vwind_impl_wgt=vwind_impl_wgt,
        hmask_dd3d=hmask_dd3d,
        scalfac_dd3d=scalfac_dd3d,
        coeff1_dwdz=coeff1_dwdz,
        coeff2_dwdz=coeff2_dwdz,
        coeff_gradekin=coeff_gradekin,
        c_owner_mask=c_owner_mask,
        cell_center_lat=cell_center_lat,
        cell_center_lon=cell_center_lon,
        edge_center_lat=edge_center_lat,
        edge_center_lon=edge_center_lon,
        primal_normal_x=primal_normal_x,
        primal_normal_y=primal_normal_y,
        rayleigh_damping_height=rayleigh_damping_height,
        itime_scheme=itime_scheme,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        ndyn_substeps=ndyn_substeps,
        rayleigh_type=rayleigh_type,
        rayleigh_coeff=rayleigh_coeff,
        divdamp_order=divdamp_order,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        divdamp_type=divdamp_type,
        divdamp_trans_start=divdamp_trans_start,
        divdamp_trans_end=divdamp_trans_end,
        l_vert_nested=l_vert_nested,
        rhotheta_offctr=rhotheta_offctr,
        veladv_offctr=veladv_offctr,
        max_nudging_coeff=max_nudging_coeff,
        divdamp_fac=divdamp_fac,
        divdamp_fac2=divdamp_fac2,
        divdamp_fac3=divdamp_fac3,
        divdamp_fac4=divdamp_fac4,
        divdamp_z=divdamp_z,
        divdamp_z2=divdamp_z2,
        divdamp_z3=divdamp_z3,
        divdamp_z4=divdamp_z4,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        nflat_gradp=nflat_gradp,
        num_levels=num_levels,
    )

    # solve nh run parameters
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")
    initial_divdamp_fac = sp.divdamp_fac_o2()

    # PrepAdvection
    vn_traj = sp.vn_traj()
    mass_flx_me = sp.mass_flx_me()
    mass_flx_ic = sp.mass_flx_ic()

    # Diagnostic state parameters
    theta_v_ic = sp.theta_v_ic()
    exner_pr = sp.exner_pr()
    rho_ic = sp.rho_ic()
    ddt_exner_phy = sp.ddt_exner_phy()
    grf_tend_rho = sp.grf_tend_rho()
    grf_tend_thv = sp.grf_tend_thv()
    grf_tend_w = sp.grf_tend_w()
    mass_fl_e = sp.mass_fl_e()
    ddt_vn_phy = sp.ddt_vn_phy()
    grf_tend_vn = sp.grf_tend_vn()
    ddt_vn_apc_ntl1 = sp.ddt_vn_apc_pc(1)
    ddt_vn_apc_ntl2 = sp.ddt_vn_apc_pc(2)
    ddt_w_adv_ntl1 = sp.ddt_w_adv_pc(1)
    ddt_w_adv_ntl2 = sp.ddt_w_adv_pc(2)
    vt = sp.vt()
    vn_ie = sp.vn_ie()
    w_concorr_c = sp.w_concorr_c()
    exner_dyn_incr = sp.exner_dyn_incr()

    # Prognostic state parameters
    w_now = sp.w_now()
    vn_now = sp.vn_now()
    theta_v_now = sp.theta_v_now()
    rho_now = sp.rho_now()
    exner_now = sp.exner_now()

    w_new = sp.w_new()
    vn_new = sp.vn_new()
    theta_v_new = sp.theta_v_new()
    rho_new = sp.rho_new()
    exner_new = sp.exner_new()

    # use fortran indices (also in the driving loop to compute i_substep)
    nnow = 1
    nnew = 2

    for i_substep in range(1, ndyn_substeps + 1):
        is_last_substep = i_substep == (ndyn_substeps)

        dycore_wrapper.solve_nh_run(
            rho_now=rho_now,
            rho_new=rho_new,
            exner_now=exner_now,
            exner_new=exner_new,
            w_now=w_now,
            w_new=w_new,
            theta_v_now=theta_v_now,
            theta_v_new=theta_v_new,
            vn_now=vn_now,
            vn_new=vn_new,
            w_concorr_c=w_concorr_c,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_w_adv_ntl1=ddt_w_adv_ntl1,
            ddt_w_adv_ntl2=ddt_w_adv_ntl2,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            exner_pr=exner_pr,
            exner_dyn_incr=exner_dyn_incr,
            ddt_exner_phy=ddt_exner_phy,
            grf_tend_rho=grf_tend_rho,
            grf_tend_thv=grf_tend_thv,
            grf_tend_w=grf_tend_w,
            mass_fl_e=mass_fl_e,
            ddt_vn_phy=ddt_vn_phy,
            grf_tend_vn=grf_tend_vn,
            vn_ie=vn_ie,
            vt=vt,
            mass_flx_me=mass_flx_me,
            mass_flx_ic=mass_flx_ic,
            vn_traj=vn_traj,
            dtime=dtime,
            lprep_adv=lprep_adv,
            clean_mflx=clean_mflx,
            recompute=recompute,
            linit=linit,
            divdamp_fac_o2=initial_divdamp_fac,
            ndyn_substeps=ndyn_substeps,
            idyn_timestep=i_substep,
            nnow=nnow,
            nnew=nnew,
        )
        linit = False
        recompute = False
        clean_mflx = False
        if not is_last_substep:
            ntemp = nnow
            nnow = nnew
            nnew = ntemp

    cell_start_lb_plus2 = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    edge_start_lb_plus4 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )

    assert helpers.dallclose(
        rho_ic.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        theta_v_ic.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        mass_fl_e.asnumpy()[edge_start_lb_plus4:, :],
        savepoint_nonhydro_exit.mass_fl_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=5e-7,
    )

    assert helpers.dallclose(
        mass_flx_me.asnumpy(),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        atol=5e-7,
    )

    assert helpers.dallclose(
        vn_traj.asnumpy(),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        atol=1e-12,
    )

    # we compare against _now fields as _new and _now are switched internally in the granule.
    assert helpers.dallclose(
        theta_v_now.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
        atol=5e-7,
    )

    assert helpers.dallclose(
        rho_now.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        exner_now.asnumpy(),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        w_now.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        vn_now.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        atol=5e-13,
    )
    assert helpers.dallclose(
        exner_dyn_incr.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )
