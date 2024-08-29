# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import pytest
from icon4pytools.py2fgen.wrappers.dycore import grid_init, solve_nh_init, solve_nh_run

from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.dycore.nh_solve.solve_nonhydro import (
    DivergenceDampingOrder,
    HorizontalPressureDiscretizationType,
    RhoThetaAdvectionType,
    TimeSteppingScheme,
)
from icon4py.model.atmosphere.dycore.state_utils import (
    states as solve_nh_states,
    utils as solve_nh_utils,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.settings import backend
from icon4py.model.common.states import prognostic_state as prognostics
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
    serialbox_utils as sb,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from .utils import (
    construct_config,
    construct_interpolation_state_for_nonhydro,
    construct_nh_metric_state,
)


@pytest.mark.datatest
def test_validate_divdamp_fields_against_savepoint_values(
    grid_savepoint,
    savepoint_nonhydro_init,
    icon_grid,
):
    config = solve_nh.NonHydrostaticConfig()
    divdamp_fac_o2 = 0.032
    mean_cell_area = grid_savepoint.mean_cell_area()
    enh_divdamp_fac = field_alloc.allocate_zero_field(KDim, grid=icon_grid, is_halfdim=False)
    scal_divdamp = field_alloc.allocate_zero_field(KDim, grid=icon_grid, is_halfdim=False)
    bdy_divdamp = field_alloc.allocate_zero_field(KDim, grid=icon_grid, is_halfdim=False)
    smagorinsky.en_smag_fac_for_zero_nshift.with_backend(backend)(
        grid_savepoint.vct_a(),
        config.divdamp_fac,
        config.divdamp_fac2,
        config.divdamp_fac3,
        config.divdamp_fac4,
        config.divdamp_z,
        config.divdamp_z2,
        config.divdamp_z3,
        config.divdamp_z4,
        enh_divdamp_fac,
        offset_provider={"Koff": KDim},
    )
    solve_nh_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_order=config.divdamp_order,
        mean_cell_area=mean_cell_area,
        divdamp_fac_o2=divdamp_fac_o2,
        out=scal_divdamp,
        offset_provider={},
    )
    solve_nh_utils._calculate_bdy_divdamp.with_backend(backend)(
        scal_divdamp,
        config.nudge_max_coeff,
        constants.DBL_EPS,
        out=bdy_divdamp,
        offset_provider={},
    )

    assert helpers.dallclose(
        scal_divdamp.asnumpy(), savepoint_nonhydro_init.scal_divdamp().asnumpy()
    )
    assert helpers.dallclose(bdy_divdamp.asnumpy(), savepoint_nonhydro_init.bdy_divdamp().asnumpy())


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(1, 1)])
@pytest.mark.parametrize(
    "experiment,step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_nonhydro_predictor_step(
    istep_init,
    istep_exit,
    jstep_init,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    experiment,
    ndyn_substeps,
    caplog,
):
    caplog.set_level(logging.DEBUG)
    config = construct_config(experiment, ndyn_substeps)
    sp = savepoint_nonhydro_init
    sp_exit = savepoint_nonhydro_exit
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    nnow = 0
    nnew = 1

    diagnostic_state_nh = construct_diagnostics(sp)

    interpolation_state = construct_interpolation_state_for_nonhydro(interpolation_savepoint)
    metric_state_nonhydro = construct_nh_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: h_grid.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: h_grid.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro()
    nlev = icon_grid.num_levels
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)
    solve_nonhydro.set_timelevels(nnow, nnew)
    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        z_fields=solve_nonhydro.intermediate_fields,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        at_first_substep=(jstep_init == 0),
        nnow=nnow,
        nnew=nnew,
    )

    cell_start_lb_plus2 = icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim) + 2
    )
    cell_start_nudging = icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.nudging(CellDim)
    )
    edge_start_lb_plus4 = icon_grid.get_start_index(
        EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
    )
    edge_start_lb_plus6 = icon_grid.get_start_index(
        EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6
    )
    edge_start_nuding_plus1 = icon_grid.get_start_index(
        EdgeDim, h_grid.HorizontalMarkerIndex.nudging(EdgeDim) + 1
    )

    # stencils 2, 3
    assert helpers.dallclose(
        diagnostic_state_nh.exner_pr.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.exner_pr().asnumpy()[cell_start_lb_plus2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ex_pr.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_exner_ex_pr().asnumpy()[cell_start_lb_plus2:, :],
    )

    # stencils 4,5
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ic.asnumpy()[cell_start_lb_plus2:, nlev - 1],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lb_plus2:, nlev - 1],
    )
    nflatlev = vertical_params.nflatlev
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ic.asnumpy()[cell_start_lb_plus2:, nflatlev : nlev - 1],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lb_plus2:, nflatlev : nlev - 1],
        rtol=1.0e-9,
    )
    # stencil 6
    assert helpers.dallclose(
        solve_nonhydro.z_dexner_dz_c_1.asnumpy()[cell_start_lb_plus2:, nflatlev:],
        sp_exit.z_dexner_dz_c(1).asnumpy()[cell_start_lb_plus2:, nflatlev:],
        atol=5e-18,
    )

    # stencils 7,8,9
    assert helpers.dallclose(
        diagnostic_state_nh.rho_ic.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_th_ddz_exner_c.asnumpy()[cell_start_lb_plus2:, 1:],
        sp_exit.z_th_ddz_exner_c().asnumpy()[cell_start_lb_plus2:, 1:],
    )

    # stencils 7,8,9, 11
    assert helpers.dallclose(
        solve_nonhydro.z_theta_v_pr_ic.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_theta_v_pr_ic().asnumpy()[cell_start_lb_plus2:, :],
    )
    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_ic.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
    )
    # stencils 7,8,9, 13
    assert helpers.dallclose(
        solve_nonhydro.z_rth_pr_1.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_rth_pr(1).asnumpy()[cell_start_lb_plus2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_rth_pr_2.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_rth_pr(2).asnumpy()[cell_start_lb_plus2:, :],
    )

    # stencils 12
    nflat_gradp = vertical_params.nflat_gradp
    assert helpers.dallclose(
        solve_nonhydro.z_dexner_dz_c_2.asnumpy()[cell_start_lb_plus2:, nflat_gradp:],
        sp_exit.z_dexner_dz_c(2).asnumpy()[cell_start_lb_plus2:, nflat_gradp:],
        atol=1e-22,
    )

    # grad_green_gauss_cell_dsl
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_1.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_grad_rth(1).asnumpy()[cell_start_lb_plus2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_2.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_grad_rth(2).asnumpy()[cell_start_lb_plus2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_3.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_grad_rth(3).asnumpy()[cell_start_lb_plus2:, :],
        rtol=5e-6,
        atol=1e-17,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_4.asnumpy()[cell_start_lb_plus2:, :],
        sp_exit.z_grad_rth(4).asnumpy()[cell_start_lb_plus2:, :],
        rtol=1e-6,
        atol=1e-21,
    )

    # compute_horizontal_advection_of_rho_and_theta
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_rho_e.asnumpy()[edge_start_lb_plus6:, :],
        sp_exit.z_rho_e().asnumpy()[edge_start_lb_plus6:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_theta_v_e.asnumpy()[edge_start_lb_plus6:, :],
        sp_exit.z_theta_v_e().asnumpy()[edge_start_lb_plus6:, :],
    )

    # stencils 18,19, 20, 22
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_gradh_exner.asnumpy()[edge_start_nuding_plus1:, :],
        sp_exit.z_gradh_exner().asnumpy()[edge_start_nuding_plus1:, :],
        atol=1e-20,
    )
    # stencil 21
    assert helpers.dallclose(
        solve_nonhydro.z_hydro_corr.asnumpy()[edge_start_nuding_plus1:, nlev - 1],
        sp_exit.z_hydro_corr().asnumpy()[edge_start_nuding_plus1:, nlev - 1],
        atol=1e-20,
    )
    prognostic_state_nnew = prognostic_state_ls[1]
    vn_new_reference = sp_exit.vn_new().asnumpy()

    # stencils 24
    assert helpers.dallclose(
        prognostic_state_nnew.vn.asnumpy()[edge_start_nuding_plus1:, :],
        vn_new_reference[edge_start_nuding_plus1:, :],
        atol=6e-15,
    )
    # stencil 29
    assert helpers.dallclose(
        prognostic_state_nnew.vn.asnumpy()[:edge_start_nuding_plus1, :],
        vn_new_reference[:edge_start_nuding_plus1, :],
    )

    # stencil 30
    assert helpers.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy(),
        sp_exit.z_vn_avg().asnumpy(),
        atol=5e-14,
    )
    # stencil 30
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_graddiv_vn.asnumpy()[edge_start_lb_plus4:, :],
        sp_exit.z_graddiv_vn().asnumpy()[edge_start_lb_plus4:, :],
        atol=5e-20,
    )
    # stencil 30
    assert helpers.dallclose(
        diagnostic_state_nh.vt.asnumpy(),
        sp_exit.vt().asnumpy(),
        atol=5e-14,
    )

    # stencil 32
    assert helpers.dallclose(
        diagnostic_state_nh.mass_fl_e.asnumpy(),
        sp_exit.mass_fl_e().asnumpy(),
        atol=4e-12,
    )
    # stencil 32
    # TODO: @abishekg7 higher tol.
    assert helpers.dallclose(
        solve_nonhydro.z_theta_v_fl_e.asnumpy()[edge_start_lb_plus4:, :],
        sp_exit.z_theta_v_fl_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=1e-9,
    )

    # stencil 35,36, 37,38
    assert helpers.dallclose(
        diagnostic_state_nh.vn_ie.asnumpy()[edge_start_lb_plus4:, :],
        sp_exit.vn_ie().asnumpy()[edge_start_lb_plus4:, :],
        atol=2e-14,
    )

    # stencil 35,36, 37,38
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_vt_ie.asnumpy()[edge_start_lb_plus4:, :],
        sp_exit.z_vt_ie().asnumpy()[edge_start_lb_plus4:, :],
        atol=2e-14,
    )
    # stencil 35,36
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_kin_hor_e.asnumpy()[edge_start_lb_plus4:, :],
        sp_exit.z_kin_hor_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=1e-20,
    )
    # stencil 35
    assert helpers.dallclose(
        solve_nonhydro.z_w_concorr_me.asnumpy()[edge_start_lb_plus4:, nflatlev:],
        sp_exit.z_w_concorr_me().asnumpy()[edge_start_lb_plus4:, nflatlev:],
        atol=1e-15,
    )

    # stencils 39,40
    assert helpers.dallclose(
        diagnostic_state_nh.w_concorr_c.asnumpy(),
        sp_exit.w_concorr_c().asnumpy(),
        atol=1e-15,
    )

    # stencil 41
    assert helpers.dallclose(
        solve_nonhydro.z_flxdiv_mass.asnumpy(),
        sp_exit.z_flxdiv_mass().asnumpy(),
        atol=5e-13,  # TODO (magdalena) was 5e-15 for local experiment only
    )

    # TODO: @abishekg7 higher tol.
    assert helpers.dallclose(
        solve_nonhydro.z_flxdiv_theta.asnumpy(),
        sp_exit.z_flxdiv_theta().asnumpy(),
        atol=5e-12,
    )

    # stencils 43, 46, 47
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_contr_w_fl_l.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_contr_w_fl_l().asnumpy()[cell_start_nudging:, :],
        atol=2e-15,
    )

    # stencil 43
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_w_expl.asnumpy()[cell_start_nudging:, 1:nlev],
        sp_exit.z_w_expl().asnumpy()[cell_start_nudging:, 1:nlev],
        atol=1e-14,
    )

    # stencil 44, 45
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_alpha.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_alpha().asnumpy()[cell_start_nudging:, :],
        atol=5e-13,
    )
    # stencil 44
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_beta.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_beta().asnumpy()[cell_start_nudging:, :],
        atol=2e-15,
    )
    # stencil 45_b, 52
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_q.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_q().asnumpy()[cell_start_nudging:, :],
        atol=2e-15,
    )
    # stencil 48, 49
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_rho_expl.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_rho_expl().asnumpy()[cell_start_nudging:, :],
        atol=2e-15,
    )
    # stencil 48, 49
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_exner_expl.asnumpy()[cell_start_nudging:, :],
        sp_exit.z_exner_expl().asnumpy()[cell_start_nudging:, :],
        atol=2e-15,
    )

    # end
    assert helpers.dallclose(prognostic_state_nnew.rho.asnumpy(), sp_exit.rho_new().asnumpy())
    assert helpers.dallclose(
        prognostic_state_nnew.w.asnumpy(), sp_exit.w_new().asnumpy(), atol=7e-14
    )

    assert helpers.dallclose(prognostic_state_nnew.exner.asnumpy(), sp_exit.exner_new().asnumpy())
    assert helpers.dallclose(
        prognostic_state_nnew.theta_v.asnumpy(), sp_exit.theta_v_new().asnumpy()
    )


def construct_diagnostics(init_savepoint: sb.IconNonHydroInitSavepoint):
    return solve_nh_states.DiagnosticStateNonHydro(
        theta_v_ic=init_savepoint.theta_v_ic(),
        exner_pr=init_savepoint.exner_pr(),
        rho_ic=init_savepoint.rho_ic(),
        ddt_exner_phy=init_savepoint.ddt_exner_phy(),
        grf_tend_rho=init_savepoint.grf_tend_rho(),
        grf_tend_thv=init_savepoint.grf_tend_thv(),
        grf_tend_w=init_savepoint.grf_tend_w(),
        mass_fl_e=init_savepoint.mass_fl_e(),
        ddt_vn_phy=init_savepoint.ddt_vn_phy(),
        grf_tend_vn=init_savepoint.grf_tend_vn(),
        ddt_vn_apc_ntl1=init_savepoint.ddt_vn_apc_pc(1),
        ddt_vn_apc_ntl2=init_savepoint.ddt_vn_apc_pc(2),
        ddt_w_adv_ntl1=init_savepoint.ddt_w_adv_pc(1),
        ddt_w_adv_ntl2=init_savepoint.ddt_w_adv_pc(2),
        vt=init_savepoint.vt(),
        vn_ie=init_savepoint.vn_ie(),
        w_concorr_c=init_savepoint.w_concorr_c(),
        rho_incr=None,  # sp.rho_incr(),
        vn_incr=None,  # sp.vn_incr(),
        exner_incr=None,  # sp.exner_incr(),
        exner_dyn_incr=init_savepoint.exner_dyn_incr(),
    )


def create_vertical_params(vertical_config, grid_savepoint):
    return v_grid.VerticalGridParams(
        vertical_config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )


@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit", [(2, 2)])
@pytest.mark.parametrize(
    "experiment,step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (
            dt_utils.GLOBAL_EXPERIMENT,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_nonhydro_corrector_step(
    istep_init,
    istep_exit,
    jstep_init,
    step_date_init,
    step_date_exit,
    icon_grid,
    savepoint_nonhydro_init,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    experiment,
    ndyn_substeps,
    caplog,
):
    caplog.set_level(logging.DEBUG)
    config = construct_config(experiment, ndyn_substeps)
    sp = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
    )

    nnow = 0
    nnew = 1

    diagnostic_state_nh = construct_diagnostics(sp)

    z_fields = solve_nh.IntermediateFields(
        z_gradh_exner=sp.z_gradh_exner(),
        z_alpha=sp.z_alpha(),
        z_beta=sp.z_beta(),
        z_w_expl=sp.z_w_expl(),
        z_exner_expl=sp.z_exner_expl(),
        z_q=sp.z_q(),
        z_contr_w_fl_l=sp.z_contr_w_fl_l(),
        z_rho_e=sp.z_rho_e(),
        z_theta_v_e=sp.z_theta_v_e(),
        z_graddiv_vn=sp.z_graddiv_vn(),
        z_rho_expl=sp.z_rho_expl(),
        z_dwdz_dd=sp.z_dwdz_dd(),
        z_kin_hor_e=sp.z_kin_hor_e(),
        z_vt_ie=sp.z_vt_ie(),
    )

    divdamp_fac_o2 = sp.divdamp_fac_o2()

    interpolation_state = construct_interpolation_state_for_nonhydro(interpolation_savepoint)
    metric_state_nonhydro = construct_nh_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: h_grid.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: h_grid.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)
    solve_nonhydro.set_timelevels(nnow, nnew)

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_ls,
        z_fields=z_fields,
        prep_adv=prep_adv,
        divdamp_fac_o2=divdamp_fac_o2,
        dtime=dtime,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    if icon_grid.limited_area:
        assert helpers.dallclose(solve_nonhydro._bdy_divdamp.asnumpy(), sp.bdy_divdamp().asnumpy())

    assert helpers.dallclose(solve_nonhydro.scal_divdamp.asnumpy(), sp.scal_divdamp().asnumpy())
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state_nh.rho_ic.asnumpy(),
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_ic.asnumpy(),
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
    )

    # stencil 17
    assert helpers.dallclose(
        z_fields.z_graddiv_vn.asnumpy(),
        savepoint_nonhydro_exit.z_graddiv_vn().asnumpy(),
        atol=1e-12,
    )

    # stencil 23,26, 27, 4th_order_divdamp
    assert helpers.dallclose(
        prognostic_state_ls[nnew].vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-9,  # TODO (magdalena) was 1e-10 for local experiment only
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].exner.asnumpy(),
        savepoint_nonhydro_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].theta_v.asnumpy(),
        savepoint_nonhydro_exit.theta_v_new().asnumpy(),
    )
    # stencil 31
    assert helpers.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy(),
        savepoint_nonhydro_exit.z_vn_avg().asnumpy(),
        rtol=5e-7,
    )

    # stencil 32
    assert helpers.dallclose(
        diagnostic_state_nh.mass_fl_e.asnumpy(),
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        rtol=5e-7,  # TODO (magdalena) was rtol=1e-10 for local experiment only
    )

    # stencil 33, 34
    assert helpers.dallclose(
        prep_adv.mass_flx_me.asnumpy(),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        rtol=5e-7,  # TODO (magdalena) was rtol=1e-10 for local experiment only
    )
    # stencil 33, 34
    assert helpers.dallclose(
        prep_adv.vn_traj.asnumpy(),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        rtol=5e-7,  # TODO (magdalena) was rtol=1e-10 for local experiment only
    )
    # stencil 60 only relevant for last substep
    assert helpers.dallclose(
        diagnostic_state_nh.exner_dyn_incr.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


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
        (
            dt_utils.GLOBAL_EXPERIMENT,
            "2000-01-01T00:00:02.000",
            "2000-01-01T00:00:02.000",
        ),
    ],
)
def test_run_solve_nonhydro_single_step(
    istep_init,
    istep_exit,
    jstep_init,
    jstep_exit,
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    icon_grid,
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
):
    caplog.set_level(logging.DEBUG)
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)

    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
    )

    nnow = 0
    nnew = 1
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = construct_diagnostics(sp)

    interpolation_state = construct_interpolation_state_for_nonhydro(interpolation_savepoint)
    metric_state_nonhydro = construct_nh_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: h_grid.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: h_grid.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    prognostic_state_ls = create_prognostic_states(sp)

    initial_divdamp_fac = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_ls=prognostic_state_ls,
        prep_adv=prep_adv,
        divdamp_fac_o2=initial_divdamp_fac,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        nnew=nnew,
        nnow=nnow,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_first_substep=jstep_init == 0,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    prognostic_state_nnew = prognostic_state_ls[1]
    assert helpers.dallclose(
        prognostic_state_nnew.theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_nnew.exner.asnumpy(), sp_step_exit.exner_new().asnumpy()
    )

    assert helpers.dallclose(
        prognostic_state_nnew.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-12,
        atol=1e-13,
    )

    assert helpers.dallclose(
        prognostic_state_nnew.rho.asnumpy(), savepoint_nonhydro_exit.rho_new().asnumpy()
    )

    assert helpers.dallclose(
        prognostic_state_nnew.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        diagnostic_state_nh.exner_dyn_incr.asnumpy(),
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
def test_run_solve_nonhydro_multi_step(
    step_date_init,
    step_date_exit,
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
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(CellDim, KDim, grid=icon_grid),
    )

    nnow = 0
    nnew = 1
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = construct_diagnostics(sp)
    prognostic_state_ls = create_prognostic_states(sp)

    interpolation_state = construct_interpolation_state_for_nonhydro(interpolation_savepoint)
    metric_state_nonhydro = construct_nh_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: h_grid.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: h_grid.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro()
    solve_nonhydro.init(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
    )

    for i_substep in range(ndyn_substeps):
        is_first_substep = i_substep == 0
        is_last_substep = i_substep == (ndyn_substeps - 1)
        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state_ls=prognostic_state_ls,
            prep_adv=prep_adv,
            divdamp_fac_o2=sp.divdamp_fac_o2(),
            dtime=dtime,
            l_recompute=recompute,
            l_init=linit,
            nnew=nnew,
            nnow=nnow,
            lclean_mflx=clean_mflx,
            lprep_adv=lprep_adv,
            at_first_substep=is_first_substep,
            at_last_substep=is_last_substep,
        )
        linit = False
        recompute = False
        clean_mflx = False
        if not is_last_substep:
            ntemp = nnow
            nnow = nnew
            nnew = ntemp

    cell_start_lb_plus2 = icon_grid.get_start_index(
        CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(CellDim) + 2
    )
    edge_start_lb_plus4 = icon_grid.get_start_index(
        EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4
    )

    assert helpers.dallclose(
        diagnostic_state_nh.rho_ic.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_ic.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_graddiv_vn.asnumpy()[edge_start_lb_plus4:, :],
        savepoint_nonhydro_exit.z_graddiv_vn().asnumpy()[edge_start_lb_plus4:, :],
        atol=1.0e-18,
    )

    assert helpers.dallclose(
        diagnostic_state_nh.mass_fl_e.asnumpy()[edge_start_lb_plus4:, :],
        savepoint_nonhydro_exit.mass_fl_e().asnumpy()[edge_start_lb_plus4:, :],
        atol=5e-7,
    )

    assert helpers.dallclose(
        prep_adv.mass_flx_me.asnumpy(),
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        atol=5e-7,
    )

    assert helpers.dallclose(
        prep_adv.vn_traj.asnumpy(),
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        atol=1e-12,
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].exner.asnumpy(),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_state_ls[nnew].vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        atol=5e-13,
    )
    assert helpers.dallclose(
        diagnostic_state_nh.exner_dyn_incr.asnumpy(),
        savepoint_nonhydro_exit.exner_dyn_incr().asnumpy(),
        atol=1e-14,
    )


@pytest.mark.datatest
def test_non_hydrostatic_params(savepoint_nonhydro_init):
    config = solve_nh.NonHydrostaticConfig()
    params = solve_nh.NonHydrostaticParams(config)

    assert params.wgt_nnew_vel == savepoint_nonhydro_init.wgt_nnew_vel()
    assert params.wgt_nnow_vel == savepoint_nonhydro_init.wgt_nnow_vel()
    assert params.wgt_nnew_rth == savepoint_nonhydro_init.wgt_nnew_rth()
    assert params.wgt_nnow_rth == savepoint_nonhydro_init.wgt_nnow_rth()


def create_prognostic_states(sp):
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
    prognostic_state_ls = [prognostic_state_nnow, prognostic_state_nnew]
    return prognostic_state_ls


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
):
    caplog.set_level(logging.DEBUG)

    # savepoints
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_exit

    # non hydrostatic config parameters
    itime_scheme = TimeSteppingScheme.MOST_EFFICIENT
    iadv_rhotheta = RhoThetaAdvectionType.MIURA
    igradp_method = HorizontalPressureDiscretizationType.TAYLOR_HYDRO
    ndyn_substeps = ndyn_substeps
    rayleigh_type = constants.RayleighType.KLEMP
    rayleigh_coeff = 0.05
    divdamp_order = DivergenceDampingOrder.COMBINED
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

    # grid params
    cell_starts = grid_savepoint.cells_start_index()
    cell_ends = grid_savepoint.cells_end_index()
    vertex_starts = grid_savepoint.vertex_start_index()
    vertex_ends = grid_savepoint.vertex_end_index()
    edge_starts = grid_savepoint.edge_start_index()
    edge_ends = grid_savepoint.edge_end_index()
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")
    c2e = grid_savepoint.c2e()
    e2c = grid_savepoint.e2c()
    e2v = grid_savepoint.e2v()
    v2e = grid_savepoint.v2e()
    v2c = grid_savepoint.v2c()
    e2c2v = grid_savepoint.e2c2v()
    c2v = grid_savepoint.c2v()
    c2e2c = grid_savepoint.c2e2c()
    e2c2e = grid_savepoint.e2c2e()
    c2e2c2e = grid_savepoint.c2e2c2e()

    # global grid params
    global_root = 4
    global_level = 9

    grid_init(
        cell_starts=cell_starts,
        cell_ends=cell_ends,
        vertex_starts=vertex_starts,
        vertex_ends=vertex_ends,
        edge_starts=edge_starts,
        edge_ends=edge_ends,
        c2e=c2e,
        e2c=e2c,
        c2e2c=c2e2c,
        c2e2c2e=c2e2c2e,
        e2c2e=e2c2e,
        e2v=e2v,
        v2e=v2e,
        v2c=v2c,
        e2c2v=e2c2v,
        c2v=c2v,
        global_root=global_root,
        global_level=global_level,
        num_vertices=num_vertices,
        num_cells=num_cells,
        num_edges=num_edges,
        vertical_size=vertical_size,
        limited_area=limited_area,
    )

    # call solve init
    solve_nh_init(
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

    solve_nh_run(
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
        idyn_timestep=jstep_init,
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
