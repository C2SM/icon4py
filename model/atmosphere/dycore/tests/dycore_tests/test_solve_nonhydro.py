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

import icon4py.model.common.grid.states as grid_states
from icon4py.model.atmosphere.dycore import (
    dycore_states,
    dycore_utils,
    fused_solve_nonhydro_stencil_15_to_28,
    solve_nonhydro as solve_nh,
)
from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import horizontal as h_grid, vertical as v_grid
from icon4py.model.common.math import smagorinsky
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    helpers,
)

from . import utils


@pytest.mark.datatest
def test_validate_divdamp_fields_against_savepoint_values(
    grid_savepoint, savepoint_nonhydro_init, icon_grid, backend
):
    config = solve_nh.NonHydrostaticConfig()
    divdamp_fac_o2 = 0.032
    mean_cell_area = grid_savepoint.mean_cell_area()
    enh_divdamp_fac = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
    scal_divdamp = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
    bdy_divdamp = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
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
    dycore_utils._calculate_scal_divdamp.with_backend(backend)(
        enh_divdamp_fac=enh_divdamp_fac,
        divdamp_order=config.divdamp_order,
        mean_cell_area=mean_cell_area,
        divdamp_fac_o2=divdamp_fac_o2,
        out=scal_divdamp,
        offset_provider={},
    )
    dycore_utils._calculate_bdy_divdamp.with_backend(backend)(
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
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize(
    "istep_init, step_date_init, substep_init, at_initial_timestep",
    [
        (1, "2021-06-20T12:00:10.000", 1, True),
        (2, "2021-06-20T12:00:10.000", 1, True),
        (1, "2021-06-20T12:00:10.000", 2, True),
        (2, "2021-06-20T12:00:10.000", 2, True),
        (1, "2021-06-20T12:00:20.000", 1, False),
        (2, "2021-06-20T12:00:20.000", 1, False),
        (1, "2021-06-20T12:00:20.000", 2, False),
        (2, "2021-06-20T12:00:20.000", 2, False),
    ],
)
def test_time_step_flags(
    experiment,
    istep_init,
    substep_init,
    step_date_init,
    at_initial_timestep,
    savepoint_nonhydro_init,
):
    sp = savepoint_nonhydro_init

    recompute = sp.get_metadata("recompute").get("recompute")
    clean_mflx = sp.get_metadata("clean_mflx").get("clean_mflx")
    linit = sp.get_metadata("linit").get("linit")

    assert recompute == (substep_init == 1)
    assert clean_mflx == (substep_init == 1)
    assert linit == (at_initial_timestep and (substep_init == 1))


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 1, 1, True)]
)
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
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
    substep_init,
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
    at_initial_timestep,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
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

    diagnostic_state_nh = utils.construct_diagnostics(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

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
    at_first_substep = substep_init == 1

    prognostic_states = utils.create_prognostic_states(sp)

    if not (at_initial_timestep and at_first_substep):
        diagnostic_state_nh.ddt_w_adv_pc.swap()
    if not at_first_substep:
        diagnostic_state_nh.ddt_vn_apc_pc.swap()

    solve_nonhydro.run_predictor_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        z_fields=solve_nonhydro.intermediate_fields,
        dtime=dtime,
        at_initial_timestep=at_initial_timestep,
        at_first_substep=at_first_substep,
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
        sp_exit.z_dexner_dz_c(0).asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev:],
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
        rtol=2.0e-12,
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
        sp_exit.z_rth_pr(0).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.z_rth_pr_2.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_rth_pr(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 12
    nflat_gradp = vertical_params.nflat_gradp
    assert helpers.dallclose(
        solve_nonhydro.z_dexner_dz_c_2.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflat_gradp:
        ],
        sp_exit.z_dexner_dz_c(1).asnumpy()[cell_start_lateral_boundary_level_2:, nflat_gradp:],
        atol=1e-22,
    )

    # grad_green_gauss_cell_dsl
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_1.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(0).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_2.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_3.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(2).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=5e-6,
        atol=1e-17,
    )
    assert helpers.dallclose(
        solve_nonhydro.z_grad_rth_4.asnumpy()[cell_start_lateral_boundary_level_2:, :],
        sp_exit.z_grad_rth(3).asnumpy()[cell_start_lateral_boundary_level_2:, :],
        rtol=1e-6,
        atol=1e-21,
    )

    # TODO Fix REGIONAL
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
    prognostic_state_nnew = prognostic_states.next
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
        solve_nonhydro.z_vn_avg.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.z_vn_avg().asnumpy()[edge_start_lateral_boundary_level_5:, :],
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
        solve_nonhydro.intermediate_fields.z_q.asnumpy()[
            cell_start_nudging:, : icon_grid.num_levels
        ],
        sp_exit.z_q().asnumpy()[cell_start_nudging:, : icon_grid.num_levels],
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


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(2, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
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
    substep_init,
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
    at_initial_timestep,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    init_savepoint = savepoint_nonhydro_init
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = init_savepoint.get_metadata("dtime").get("dtime")
    lprep_adv = init_savepoint.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=init_savepoint.vn_traj(),
        mass_flx_me=init_savepoint.mass_flx_me(),
        mass_flx_ic=init_savepoint.mass_flx_ic(),
        vol_flx_ic=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend),
    )

    diagnostic_state_nh = utils.construct_diagnostics(init_savepoint)

    z_fields = solve_nh.IntermediateFields(
        z_gradh_exner=init_savepoint.z_gradh_exner(),
        z_alpha=init_savepoint.z_alpha(),
        z_beta=init_savepoint.z_beta(),
        z_w_expl=init_savepoint.z_w_expl(),
        z_exner_expl=init_savepoint.z_exner_expl(),
        z_q=init_savepoint.z_q(),
        z_contr_w_fl_l=init_savepoint.z_contr_w_fl_l(),
        z_rho_e=init_savepoint.z_rho_e(),
        z_theta_v_e=init_savepoint.z_theta_v_e(),
        z_graddiv_vn=init_savepoint.z_graddiv_vn(),
        z_rho_expl=init_savepoint.z_rho_expl(),
        z_dwdz_dd=init_savepoint.z_dwdz_dd(),
        z_kin_hor_e=init_savepoint.z_kin_hor_e(),
        z_vt_ie=init_savepoint.z_vt_ie(),
    )

    divdamp_fac_o2 = init_savepoint.divdamp_fac_o2()

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

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
    at_first_substep = substep_init == 1
    at_last_substep = substep_init == ndyn_substeps

    prognostic_states = utils.create_prognostic_states(init_savepoint)
    solve_nonhydro.update_time_levels_for_velocity_tendencies(
        diagnostic_state_nh,
        at_first_substep=at_first_substep,
        at_initial_timestep=at_initial_timestep,
    )

    if not (at_initial_timestep and at_first_substep):
        diagnostic_state_nh.ddt_w_adv_pc.swap()
    if not at_first_substep:
        diagnostic_state_nh.ddt_vn_apc_pc.swap()

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        z_fields=z_fields,
        prep_adv=prep_adv,
        divdamp_fac_o2=divdamp_fac_o2,
        dtime=dtime,
        lprep_adv=lprep_adv,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )

    if icon_grid.limited_area:
        assert helpers.dallclose(
            solve_nonhydro._bdy_divdamp.asnumpy(), init_savepoint.bdy_divdamp().asnumpy()
        )

    assert helpers.dallclose(
        solve_nonhydro.scal_divdamp.asnumpy(), init_savepoint.scal_divdamp().asnumpy()
    )
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
        prognostic_states.next.vn.asnumpy(),
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        rtol=1e-9,  # TODO (magdalena) was 1e-10 for local experiment only
    )

    assert helpers.dallclose(
        prognostic_states.next.exner.asnumpy(),
        savepoint_nonhydro_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_states.next.theta_v.asnumpy(),
        savepoint_nonhydro_exit.theta_v_new().asnumpy(),
    )
    # stencil 31 - TODO savepoint value starts differing from 0.0 at 1688 which is a n edge boundary
    assert helpers.dallclose(
        solve_nonhydro.z_vn_avg.asnumpy()[solve_nonhydro._start_edge_lateral_boundary_level_5 :, :],
        savepoint_nonhydro_exit.z_vn_avg().asnumpy()[
            solve_nonhydro._start_edge_lateral_boundary_level_5 :, :
        ],
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


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit, at_initial_timestep", [(1, 1, 2, 1, True)]
)
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
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
    substep_init,
    substep_exit,
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
    savepoint_nonhydro_step_final,
    at_initial_timestep,
    caplog,
    backend,
):
    caplog.set_level(logging.WARN)
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)

    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final
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
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend),
    )

    diagnostic_state_nh = utils.construct_diagnostics(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

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

    prognostic_states = utils.create_prognostic_states(sp)

    initial_divdamp_fac = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        divdamp_fac_o2=initial_divdamp_fac,
        dtime=dtime,
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=substep_init == 1,
        at_last_substep=substep_init == ndyn_substeps,
    )
    prognostic_state_nnew = prognostic_states.next
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


# why is this not run for APE?
@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("experiment", [dt_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize(
    "istep_init, substep_init, step_date_init, istep_exit, substep_exit, step_date_exit,  at_initial_timestep",
    [
        (1, 1, "2021-06-20T12:00:10.000", 2, 2, "2021-06-20T12:00:10.000", True),
        (1, 1, "2021-06-20T12:00:20.000", 2, 2, "2021-06-20T12:00:20.000", False),
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
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    experiment,
    ndyn_substeps,
    backend,
    at_initial_timestep,
):
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final
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
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        mass_flx_ic=sp.mass_flx_ic(),
        vol_flx_ic=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend),
    )

    linit = sp.get_metadata("linit").get("linit")

    diagnostic_state_nh = utils.construct_diagnostics(sp, swap_ddt_w_adv_pc=not linit)
    prognostic_states = utils.create_prognostic_states(sp)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

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
        at_first_substep = i_substep == 0
        at_last_substep = i_substep == (ndyn_substeps - 1)

        if not (at_initial_timestep and at_first_substep):
            diagnostic_state_nh.ddt_w_adv_pc.swap()
        if not at_first_substep:
            diagnostic_state_nh.ddt_vn_apc_pc.swap()

        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            prep_adv=prep_adv,
            divdamp_fac_o2=sp.divdamp_fac_o2(),
            dtime=dtime,
            at_initial_timestep=at_initial_timestep,
            lprep_adv=lprep_adv,
            at_first_substep=at_first_substep,
            at_last_substep=at_last_substep,
        )

        if not at_last_substep:
            prognostic_states.swap()

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
        prognostic_states.next.theta_v.asnumpy(),
        sp_step_exit.theta_v_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.next.rho.asnumpy(),
        savepoint_nonhydro_exit.rho_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.next.exner.asnumpy(),
        sp_step_exit.exner_new().asnumpy(),
    )

    assert helpers.dallclose(
        prognostic_states.next.w.asnumpy(),
        savepoint_nonhydro_exit.w_new().asnumpy(),
        atol=8e-14,
    )

    assert helpers.dallclose(
        prognostic_states.next.vn.asnumpy(),
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


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("istep_init, istep_exit, at_initial_timestep", [(1, 2, True)])
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            dt_utils.REGIONAL_EXPERIMENT,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        # (
        #     dt_utils.GLOBAL_EXPERIMENT,
        #     "2000-01-01T00:00:02.000",
        #     "2000-01-01T00:00:02.000",
        # ),
    ],
)
def test_run_solve_nonhydro_15_to_28(
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
    at_initial_timestep,
    istep_init,
    substep_init,
    substep_exit,
    savepoint_nonhydro_15_28_init,
    savepoint_nonhydro_15_28_exit,
    backend,
):
    edge_domain = h_grid.domain(dims.EdgeDim)

    start_edge_halo_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    end_edge_halo_level_2 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    start_edge_lateral_boundary = icon_grid.end_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY))
    end_edge_halo = icon_grid.end_index(edge_domain(h_grid.Zone.HALO))
    start_edge_lateral_boundary_level_7 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))
    end_edge_end = icon_grid.end_index(edge_domain(h_grid.Zone.END))

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    vert_idx = data_alloc.index_field(dim=dims.KDim, grid=icon_grid, backend=backend)
    horz_idx = data_alloc.index_field(dim=dims.EdgeDim, grid=icon_grid, backend=backend)

    p_vn = savepoint_nonhydro_15_28_init.p_vn()
    p_vt = savepoint_nonhydro_15_28_init.p_vt()
    z_rth_pr_1 = savepoint_nonhydro_15_28_init.z_rth_pr(0)
    z_rth_pr_2 = savepoint_nonhydro_15_28_init.z_rth_pr(1)
    z_exner_ex_pr = savepoint_nonhydro_15_28_init.z_exner_ex_pr()
    z_dexner_dz_c_1 = savepoint_nonhydro_15_28_init.z_dexner_dz_c(0)
    z_dexner_dz_c_2 = savepoint_nonhydro_15_28_init.z_dexner_dz_c(1)
    theta_v = savepoint_nonhydro_15_28_init.theta_v()
    theta_v_ic = savepoint_nonhydro_15_28_init.theta_v_ic()
    z_dwdz_dd = savepoint_nonhydro_15_28_init.z_dwdz_dd()
    ddt_vn_apc_ntl2 = savepoint_nonhydro_15_28_init.ddt_vn_apc_ntl(1)
    ddt_vn_apc_ntl1 = savepoint_nonhydro_15_28_init.ddt_vn_apc_ntl(0)
    ddt_vn_phy = savepoint_nonhydro_15_28_init.ddt_vn_phy()
    # vn_incr = savepoint_nonhydro_15_28_init.vn_incr()
    vn_incr = gtx.empty(
        domain={dims.EdgeDim: range(icon_grid.num_edges), dims.KDim: range(icon_grid.num_levels)}
    )
    bdy_divdamp = savepoint_nonhydro_15_28_init.bdy_divdamp()
    z_hydro_corr = savepoint_nonhydro_15_28_init.z_hydro_corr()
    z_graddiv2_vn = savepoint_nonhydro_15_28_init.z_graddiv2_vn()
    scal_divdamp = savepoint_nonhydro_15_28_init.scal_divdamp()
    z_rho_e = savepoint_nonhydro_15_28_init.z_rho_e()
    z_theta_v_e = savepoint_nonhydro_15_28_init.z_theta_v_e()
    z_gradh_exner = savepoint_nonhydro_15_28_init.z_gradh_exner()
    vn = savepoint_nonhydro_15_28_init.vn()
    z_graddiv_vn = savepoint_nonhydro_15_28_init.z_graddiv_vn()
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
    params_config = solve_nh.NonHydrostaticConfig()
    primal_normal_cell_1 = data_alloc.flatten_first_two_dims(
        dims.ECDim, field=grid_savepoint.primal_normal_cell_x()
    )
    primal_normal_cell_2 = data_alloc.flatten_first_two_dims(
        dims.ECDim, field=grid_savepoint.primal_normal_cell_y()
    )
    dual_normal_cell_1 = data_alloc.flatten_first_two_dims(
        dims.ECDim, field=grid_savepoint.dual_normal_cell_x()
    )
    dual_normal_cell_2 = data_alloc.flatten_first_two_dims(
        dims.ECDim, field=grid_savepoint.dual_normal_cell_y()
    )

    iau_wgt_dyn = params_config.iau_wgt_dyn
    itime_scheme = params_config.itime_scheme
    divdamp_order = params_config.divdamp_order
    scal_divdamp_o2 = params_config.divdamp_fac2 * grid_savepoint.mean_cell_area()
    iadv_rhotheta = params_config.iadv_rhotheta
    is_iau_active = params_config.is_iau_active
    igradp_method = params_config.igradp_method

    z_rho_e_ref = savepoint_nonhydro_15_28_exit.z_rho_e()
    z_theta_v_e_ref = savepoint_nonhydro_15_28_exit.z_theta_v_e()
    z_gradh_exner_ref = savepoint_nonhydro_15_28_exit.z_gradh_exner()
    vn_ref = savepoint_nonhydro_15_28_exit.vn()
    z_graddiv_vn_ref = savepoint_nonhydro_15_28_exit.z_graddiv_vn()

    fused_solve_nonhydro_stencil_15_to_28.fused_solve_nonhydro_stencil_15_to_28.with_backend(
        backend
    )(
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        p_vn=p_vn,
        p_vt=p_vt,
        pos_on_tplane_e_1=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_2=interpolation_savepoint.pos_on_tplane_e_y(),
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        rho_ref_me=metrics_savepoint.rho_ref_me(),
        theta_ref_me=metrics_savepoint.theta_ref_me(),
        z_rth_pr_1=z_rth_pr_1,
        z_rth_pr_2=z_rth_pr_2,
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        z_exner_ex_pr=z_exner_ex_pr,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        theta_v=theta_v,
        ikoffset=metrics_savepoint.vertoffset_gradp(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
        ipeidx_dsl=metrics_savepoint.pg_edgeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        hmask_dd3d=metrics_savepoint.hmask_dd3d(),
        scalfac_dd3d=metrics_savepoint.scalfac_dd3d(),
        z_dwdz_dd=z_dwdz_dd,
        inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
        ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
        ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
        ddt_vn_phy=ddt_vn_phy,
        vn_incr=vn_incr,
        horz_idx=horz_idx,
        vert_idx=vert_idx,
        bdy_divdamp=bdy_divdamp,
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        z_hydro_corr=z_hydro_corr,
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        z_graddiv2_vn=z_graddiv2_vn,
        scal_divdamp=scal_divdamp,
        z_rho_e=z_rho_e,
        z_theta_v_e=z_theta_v_e,
        z_gradh_exner=z_gradh_exner,
        vn=vn,
        z_graddiv_vn=z_graddiv_vn,
        divdamp_fac=config.divdamp_fac,
        divdamp_fac_o2=savepoint_nonhydro_init.divdamp_fac_o2(),
        wgt_nnow_vel=savepoint_nonhydro_init.wgt_nnow_vel(),
        wgt_nnew_vel=savepoint_nonhydro_init.wgt_nnew_vel(),
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        cpd=constants.CPD,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        itime_scheme=itime_scheme,
        p_dthalf=(0.5 * savepoint_nonhydro_init.get_metadata("dtime").get("dtime")),
        grav_o_cpd=nonhydro_params.grav_o_cpd,
        limited_area=grid_savepoint.get_metadata("limited_area").get("limited_area"),
        divdamp_order=divdamp_order,
        scal_divdamp_o2=scal_divdamp_o2,
        istep=istep_init,
        start_edge_halo_level_2=start_edge_halo_level_2,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        end_edge_halo=end_edge_halo,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        end_edge_end=end_edge_end,
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method.value,
        MIURA=solve_nh.RhoThetaAdvectionType.MIURA.value,
        TAYLOR_HYDRO=solve_nh.HorizontalPressureDiscretizationType.TAYLOR_HYDRO.value,
        nlev=icon_grid.num_levels,
        kstart_dd3d=nonhydro_params.kstart_dd3d,
        COMBINED=solve_nh.DivergenceDampingOrder.COMBINED,
        FOURTH_ORDER=solve_nh.DivergenceDampingOrder.FOURTH_ORDER,
        nflatlev=vertical_params.nflatlev,
        nflat_gradp=vertical_params.nflat_gradp,
        offset_provider={
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
            "E2EC": icon_grid.get_offset_provider("E2EC"),
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2C2EO": icon_grid.get_offset_provider("E2C2EO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(z_rho_e.asnumpy(), z_rho_e_ref.asnumpy())
    assert helpers.dallclose(z_theta_v_e.asnumpy(), z_theta_v_e_ref.asnumpy())
    assert helpers.dallclose(z_gradh_exner.asnumpy(), z_gradh_exner_ref.asnumpy())
    assert helpers.dallclose(vn.asnumpy(), vn_ref.asnumpy())
    assert helpers.dallclose(z_graddiv_vn.asnumpy(), z_graddiv_vn_ref.asnumpy())
