# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest

import icon4py.model.common.grid.geometry as geometry
from icon4py.model.atmosphere.dycore.nh_solve import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.dycore.state_utils import (
    states as solve_nh_states,
    utils as solve_nh_utils,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.settings import backend
from icon4py.model.common.test_utils import (
    datatest_utils as dt_utils,
    helpers,
)
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

from . import utils


@pytest.mark.datatest
def test_validate_divdamp_fields_against_savepoint_values(
    grid_savepoint,
    savepoint_nonhydro_init,
    icon_grid,
):
    config = solve_nh.NonHydrostaticConfig()
    divdamp_fac_o2 = 0.032
    mean_cell_area = grid_savepoint.mean_cell_area()
    enh_divdamp_fac = field_alloc.allocate_zero_field(dims.KDim, grid=icon_grid, is_halfdim=False)
    scal_divdamp = field_alloc.allocate_zero_field(dims.KDim, grid=icon_grid, is_halfdim=False)
    bdy_divdamp = field_alloc.allocate_zero_field(dims.KDim, grid=icon_grid, is_halfdim=False)
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
        offset_provider={"Koff": dims.KDim},
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
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
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
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = utils.construct_diagnostics(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: geometry.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )
    nlev = icon_grid.num_levels

    prognostic_state_swp = utils.create_prognostic_states(sp)

    solve_nonhydro.set_time_levels(diagnostic_state_nh)
    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_swp=prognostic_state_swp,
        z_fields=solve_nonhydro.intermediate_fields,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        at_first_substep=(jstep_init == 0),
    )
    cell_domain = h_grid.domain(dims.CellDim)
    edge_domain = h_grid.domain(dims.EdgeDim)

    cell_start_lateral_boundary_level_2 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    cell_start_nudging = icon_grid.start_index(cell_domain(h_grid.Zone.NUDGING))

    edge_start_lateral_boundary_level_5 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
    )
    edge_start_lateral_boundary_level_7 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    edge_start_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))

    # stencils 2, 3
    assert helpers.dallclose(
        diagnostic_state_nh.exner_pr.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.exner_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ex_pr.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_exner_ex_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 4,5
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ic.asnumpy()[cell_start_lateral_boundary_level_2:, nlev - 1],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nlev - 1],
    )
    nflatlev = vertical_params.nflatlev
    assert helpers.dallclose(
        solve_nonhydro.z_exner_ic.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1
        ],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1],
        rtol=1.0e-9,
    )
    # stencil 6
    assert helpers.dallclose(
        solve_nonhydro.z_dexner_dz_c_1.asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev:],
        sp_exit.z_dexner_dz_c(1).asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev:],
        atol=5e-18,
    )

    # stencils 7,8,9
    assert helpers.dallclose(
        diagnostic_state_nh.rho_ic.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.rho_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_th_ddz_exner_c.asnumpy()[cell_start_lateral_boundary_level_2:, 1:],
        sp_exit.z_th_ddz_exner_c().asnumpy()[cell_start_lateral_boundary_level_2:, 1:],
    )

    # stencils 7,8,9, 11
    assert helpers.dallclose(
        solve_nonhydro.z_theta_v_pr_ic.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_theta_v_pr_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_ic.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.theta_v_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    # stencils 7,8,9, 13
    assert helpers.dallclose(
        solve_nonhydro.z_rth_pr_1.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_rth_pr(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_rth_pr_2.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_rth_pr(2).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 12
    nflat_gradp = vertical_params.nflat_gradp
    assert helpers.dallclose(
        solve_nonhydro.z_dexner_dz_c_2.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflat_gradp:
        ],
        sp_exit.z_dexner_dz_c(2).asnumpy()[cell_start_lateral_boundary_level_2:, nflat_gradp:],
        atol=1e-22,
    )

    # grad_green_gauss_cell_dsl
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_1.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_2.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(2).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_3.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(3).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=5e-6,
        atol=1e-17,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_4.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(4).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )

    # compute_horizontal_advection_of_rho_and_theta
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_rho_e.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_rho_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_theta_v_e.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_theta_v_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )

    # stencils 18,19, 20, 22
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_gradh_exner.asnumpy()[edge_start_nudging_level_2:, :],
        sp_exit.z_gradh_exner().asnumpy()[edge_start_nudging_level_2:, :],
        atol=1e-20,
    )
    # stencil 21
    assert helpers.dallclose(
        solve_nonhydro.z_hydro_corr.asnumpy()[edge_start_nudging_level_2:, nlev - 1],
        sp_exit.z_hydro_corr().asnumpy()[edge_start_nudging_level_2:, nlev - 1],
        atol=1e-20,
    )
    prognostic_state_nnew = prognostic_state_swp.next
    vn_new_reference = sp_exit.vn_new().asnumpy()

    # stencils 24
    assert helpers.dallclose(
        prognostic_state_nnew.vn.asnumpy()[edge_start_nudging_level_2:, :],
        vn_new_reference[edge_start_nudging_level_2:, :],
        atol=6e-15,
    )
    # stencil 29
    assert helpers.dallclose(
        prognostic_state_nnew.vn.asnumpy()[:edge_start_nudging_level_2, :],
        vn_new_reference[:edge_start_nudging_level_2, :],
    )

    # stencil 30
    assert helpers.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy(),
        sp_exit.z_vn_avg().asnumpy(),
        atol=5e-14,
    )
    # stencil 30
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_graddiv_vn.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_graddiv_vn().asnumpy()[edge_start_lateral_boundary_level_5:, :],
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
        solve_nonhydro.z_theta_v_fl_e.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.z_theta_v_fl_e().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=1e-9,
    )

    # stencil 35,36, 37,38
    assert helpers.dallclose(
        diagnostic_state_nh.vn_ie.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.vn_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )

    # stencil 35,36, 37,38
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_vt_ie.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_vt_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )
    # stencil 35,36
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.z_kin_hor_e.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_kin_hor_e().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=1e-20,
    )
    # stencil 35
    assert helpers.dallclose(
        solve_nonhydro.z_w_concorr_me.asnumpy()[edge_start_lateral_boundary_level_5:, nflatlev:],
        sp_exit.z_w_concorr_me().asnumpy()[edge_start_lateral_boundary_level_5:, nflatlev:],
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
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    sp = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
    )

    diagnostic_state_nh = utils.construct_diagnostics(sp)

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

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: geometry.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    prognostic_state_swp = utils.create_prognostic_states(sp)
    solve_nonhydro.set_time_levels(diagnostic_state_nh)

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state=prognostic_state_swp,
        z_fields=z_fields,
        prep_adv=prep_adv,
        divdamp_fac_o2=divdamp_fac_o2,
        dtime=dtime,
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
        prognostic_state_swp.next.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-9,  # TODO (magdalena) was 1e-10 for local experiment only
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.exner.asnumpy(),
        savepoint_nonhydro_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.theta_v.asnumpy(),
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
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)

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
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
    )

    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = utils.construct_diagnostics(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: geometry.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    prognostic_state_swp = utils.create_prognostic_states(sp)

    initial_divdamp_fac = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_state_swp=prognostic_state_swp,
        prep_adv=prep_adv,
        divdamp_fac_o2=initial_divdamp_fac,
        dtime=dtime,
        l_recompute=recompute,
        l_init=linit,
        lclean_mflx=clean_mflx,
        lprep_adv=lprep_adv,
        at_first_substep=jstep_init == 0,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    prognostic_state_nnew = prognostic_state_swp.next
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
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
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
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = sp.get_metadata("dtime").get("dtime")
    lprep_adv = sp.get_metadata("prep_adv").get("prep_adv")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    prep_adv = solve_nh_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=field_alloc.allocate_zero_field(dims.CellDim, dims.KDim, grid=icon_grid),
    )

    recompute = sp.get_metadata("recompute").get("recompute")
    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = utils.construct_diagnostics(sp)
    prognostic_state_swp = utils.create_prognostic_states(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: geometry.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: geometry.EdgeParams = grid_savepoint.construct_edge_geometry()

    solve_nonhydro = solve_nh.SolveNonhydro(
        grid=icon_grid,
        config=config,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
    )

    for i_substep in range(ndyn_substeps):
        is_first_substep = i_substep == 0
        is_last_substep = i_substep == (ndyn_substeps - 1)
        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_state_swp=prognostic_state_swp,
            prep_adv=prep_adv,
            divdamp_fac_o2=sp.divdamp_fac_o2(),
            dtime=dtime,
            l_recompute=recompute,
            l_init=linit,
            lclean_mflx=clean_mflx,
            lprep_adv=lprep_adv,
            at_first_substep=is_first_substep,
            at_last_substep=is_last_substep,
        )
        linit = False
        recompute = False
        clean_mflx = False
        if not is_last_substep:
            prognostic_state_swp.swap()

    cell_start_lb_plus2 = icon_grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    edge_start_lb_plus4 = icon_grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5)
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
        prognostic_state_swp.next.theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.exner.asnumpy(),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_state_swp.next.vn.asnumpy(),
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
