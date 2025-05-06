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
    solve_nonhydro as solve_nh,
)
from icon4py.model.atmosphere.dycore.stencils import (
    compute_cell_diagnostics_for_dycore,
    compute_edge_diagnostics_for_dycore_and_update_vn,
    compute_hydrostatic_correction_term,
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
    second_order_divdamp_factor = 0.032
    mean_cell_area = grid_savepoint.mean_cell_area()
    interpolated_fourth_order_divdamp_factor = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
    fourth_order_divdamp_scaling_coeff = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
    reduced_fourth_order_divdamp_coeff_at_nest_boundary = data_alloc.zero_field(
        icon_grid,
        dims.KDim,
        backend=backend,
    )
    smagorinsky.en_smag_fac_for_zero_nshift.with_backend(backend)(
        grid_savepoint.vct_a(),
        config.fourth_order_divdamp_factor,
        config.fourth_order_divdamp_factor2,
        config.fourth_order_divdamp_factor3,
        config.fourth_order_divdamp_factor4,
        config.fourth_order_divdamp_z,
        config.fourth_order_divdamp_z2,
        config.fourth_order_divdamp_z3,
        config.fourth_order_divdamp_z4,
        interpolated_fourth_order_divdamp_factor,
        offset_provider={"Koff": dims.KDim},
    )
    dycore_utils._calculate_fourth_order_divdamp_scaling_coeff.with_backend(backend)(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        divdamp_order=config.divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        out=fourth_order_divdamp_scaling_coeff,
        offset_provider={},
    )
    dycore_utils._calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary.with_backend(
        backend
    )(
        fourth_order_divdamp_scaling_coeff,
        config.nudge_max_coeff,
        constants.DBL_EPS,
        out=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        offset_provider={},
    )

    assert helpers.dallclose(
        fourth_order_divdamp_scaling_coeff.asnumpy(),
        savepoint_nonhydro_init.scal_divdamp().asnumpy(),
    )
    assert helpers.dallclose(
        reduced_fourth_order_divdamp_coeff_at_nest_boundary.asnumpy(),
        savepoint_nonhydro_init.bdy_divdamp().asnumpy(),
    )


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
@pytest.mark.parametrize("at_initial_timestep", [True])
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

    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, backend)

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
        diagnostic_state_nh.vertical_wind_advective_tendency.swap()
    if not at_first_substep:
        diagnostic_state_nh.normal_wind_advective_tendency.swap()

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
        diagnostic_state_nh.perturbed_exner_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.exner_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.temporal_extrapolation_of_perturbed_exner.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_exner_ex_pr().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 4,5
    assert helpers.dallclose(
        solve_nonhydro.exner_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nlev - 1
        ],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nlev - 1],
    )
    nflatlev = vertical_params.nflatlev
    assert helpers.dallclose(
        solve_nonhydro.exner_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1
        ],
        sp_exit.z_exner_ic().asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev : nlev - 1],
        rtol=1.0e-9,
    )
    # stencil 6
    assert helpers.dallclose(
        solve_nonhydro.ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflatlev:
        ],
        sp_exit.z_dexner_dz_c(0).asnumpy()[cell_start_lateral_boundary_level_2:, nflatlev:],
        atol=5e-18,
    )

    # stencils 7,8,9
    assert helpers.dallclose(
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.rho_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.pressure_buoyancy_acceleration_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, 1:
        ],
        sp_exit.z_th_ddz_exner_c().asnumpy()[cell_start_lateral_boundary_level_2:, 1:],
        rtol=2.0e-12,
    )

    # stencils 7,8,9, 11
    assert helpers.dallclose(
        solve_nonhydro.perturbed_theta_v_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_theta_v_pr_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.theta_v_ic().asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    # stencils 7,8,9, 13
    assert helpers.dallclose(
        solve_nonhydro.perturbed_rho_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_rth_pr(0).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.perturbed_theta_v_at_cells_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, :
        ],
        sp_exit.z_rth_pr(1).asnumpy()[cell_start_lateral_boundary_level_2:, :],
    )

    # stencils 12
    nflat_gradp = vertical_params.nflat_gradp
    assert helpers.dallclose(
        solve_nonhydro.d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            cell_start_lateral_boundary_level_2:, nflat_gradp:
        ],
        sp_exit.z_dexner_dz_c(1).asnumpy()[cell_start_lateral_boundary_level_2:, nflat_gradp:],
        atol=1e-22,
    )

    # compute_horizontal_advection_of_rho_and_theta
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.rho_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_rho_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.theta_v_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_7:, :
        ],
        sp_exit.z_theta_v_e().asnumpy()[edge_start_lateral_boundary_level_7:, :],
    )

    # stencils 18,19, 20, 22
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.horizontal_pressure_gradient.asnumpy()[
            edge_start_nudging_level_2:, :
        ],
        sp_exit.z_gradh_exner().asnumpy()[edge_start_nudging_level_2:, :],
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
        solve_nonhydro.intermediate_fields.horizontal_gradient_of_normal_wind_divergence.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_graddiv_vn().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=5e-20,
    )
    # stencil 30
    assert helpers.dallclose(
        diagnostic_state_nh.tangential_wind.asnumpy(),
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
        diagnostic_state_nh.vn_on_half_levels.asnumpy()[edge_start_lateral_boundary_level_5:, :],
        sp_exit.vn_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )

    # stencil 35,36, 37,38
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.tangential_wind_on_half_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_vt_ie().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=2e-14,
    )
    # stencil 35,36
    assert helpers.dallclose(
        solve_nonhydro.intermediate_fields.horizontal_kinetic_energy_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, :
        ],
        sp_exit.z_kin_hor_e().asnumpy()[edge_start_lateral_boundary_level_5:, :],
        atol=1e-20,
    )
    # stencil 35
    assert helpers.dallclose(
        solve_nonhydro._contravariant_correction_at_edges_on_model_levels.asnumpy()[
            edge_start_lateral_boundary_level_5:, nflatlev:
        ],
        sp_exit.z_w_concorr_me().asnumpy()[edge_start_lateral_boundary_level_5:, nflatlev:],
        atol=1e-15,
    )

    # stencils 39,40
    assert helpers.dallclose(
        diagnostic_state_nh.contravariant_correction_at_cells_on_half_levels.asnumpy(),
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

    diagnostic_state_nh = utils.construct_diagnostics(init_savepoint, icon_grid, backend)

    z_fields = solve_nh.IntermediateFields(
        horizontal_pressure_gradient=init_savepoint.z_gradh_exner(),
        z_alpha=init_savepoint.z_alpha(),
        z_beta=init_savepoint.z_beta(),
        z_w_expl=init_savepoint.z_w_expl(),
        z_exner_expl=init_savepoint.z_exner_expl(),
        z_q=init_savepoint.z_q(),
        z_contr_w_fl_l=init_savepoint.z_contr_w_fl_l(),
        rho_at_edges_on_model_levels=init_savepoint.z_rho_e(),
        theta_v_at_edges_on_model_levels=init_savepoint.z_theta_v_e(),
        horizontal_gradient_of_normal_wind_divergence=init_savepoint.z_graddiv_vn(),
        z_rho_expl=init_savepoint.z_rho_expl(),
        dwdz_at_cells_on_model_levels=init_savepoint.z_dwdz_dd(),
        horizontal_kinetic_energy_at_edges_on_model_levels=init_savepoint.z_kin_hor_e(),
        tangential_wind_on_half_levels=init_savepoint.z_vt_ie(),
    )

    second_order_divdamp_factor = init_savepoint.divdamp_fac_o2()

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

    if not (at_initial_timestep and at_first_substep):
        diagnostic_state_nh.vertical_wind_advective_tendency.swap()
    if not at_first_substep:
        diagnostic_state_nh.normal_wind_advective_tendency.swap()

    solve_nonhydro.run_corrector_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        z_fields=z_fields,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        lprep_adv=lprep_adv,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
    )

    if icon_grid.limited_area:
        assert helpers.dallclose(
            solve_nonhydro.reduced_fourth_order_divdamp_coeff_at_nest_boundary.asnumpy(),
            init_savepoint.bdy_divdamp().asnumpy(),
        )

    assert helpers.dallclose(
        solve_nonhydro.fourth_order_divdamp_scaling_coeff.asnumpy(),
        init_savepoint.scal_divdamp().asnumpy(),
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy(),
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
    )
    # stencil 10
    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy(),
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        atol=1.0e-12,
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
    # stencil 31
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

    diagnostic_state_nh = utils.construct_diagnostics(sp, icon_grid, backend)

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

    second_order_divdamp_factor = sp.divdamp_fac_o2()
    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
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

    diagnostic_state_nh = utils.construct_diagnostics(
        sp, icon_grid, backend, swap_vertical_wind_advective_tendency=not linit
    )
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
            diagnostic_state_nh.vertical_wind_advective_tendency.swap()
        if not at_first_substep:
            diagnostic_state_nh.normal_wind_advective_tendency.swap()

        solve_nonhydro.time_step(
            diagnostic_state_nh=diagnostic_state_nh,
            prognostic_states=prognostic_states,
            prep_adv=prep_adv,
            second_order_divdamp_factor=sp.divdamp_fac_o2(),
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
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.rho_ic().asnumpy()[cell_start_lb_plus2:, :],
    )

    assert helpers.dallclose(
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy()[cell_start_lb_plus2:, :],
        savepoint_nonhydro_exit.theta_v_ic().asnumpy()[cell_start_lb_plus2:, :],
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
        atol=1e-13,
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
@pytest.mark.parametrize("at_initial_timestep", [(True)])
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
def test_compute_perturbed_quantities_and_interpolation(
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    icon_grid,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    at_initial_timestep,
    substep_init,
    substep_exit,
    savepoint_nonhydro_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_nonhydro_exit,
    backend,
):
    sp_init = savepoint_nonhydro_init
    sp_ref = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_exit = savepoint_nonhydro_exit
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)

    current_rho = sp_init.rho_now()
    current_theta_v = sp_init.theta_v_now()
    perturbed_exner_at_cells_on_model_levels = sp_init.exner_pr()
    rho_at_cells_on_half_levels = sp_init.rho_ic()
    current_exner = sp_init.exner_now()
    theta_v_at_cells_on_half_levels = sp_init.theta_v_ic()

    # local fields
    perturbed_rho_at_cells_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    perturbed_theta_v_at_cells_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
    )
    pressure_buoyancy_acceleration_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    exner_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
    )
    temporal_extrapolation_of_perturbed_exner = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
    )
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )

    limited_area = icon_grid.limited_area
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    igradp_method = config.igradp_method

    nflatlev = vertical_params.nflatlev
    nflat_gradp = vertical_params.nflat_gradp

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary = icon_grid.start_index(cell_domain(h_grid.Zone.LATERAL_BOUNDARY))
    start_cell_lateral_boundary_level_3 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    start_cell_halo_level_2 = icon_grid.start_index(cell_domain(h_grid.Zone.HALO_LEVEL_2))
    end_cell_end = icon_grid.end_index(cell_domain(h_grid.Zone.END))
    end_cell_halo = icon_grid.end_index(cell_domain(h_grid.Zone.HALO))
    end_cell_halo_level_2 = icon_grid.end_index((cell_domain(h_grid.Zone.HALO_LEVEL_2)))

    reference_rho_at_cells_on_model_levels = metrics_savepoint.rho_ref_mc()
    reference_theta_at_cells_on_model_levels = metrics_savepoint.theta_ref_mc()
    reference_theta_at_cells_on_half_levels = metrics_savepoint.theta_ref_ic()
    d2dexdz2_fac1_mc = metrics_savepoint.d2dexdz2_fac1_mc()
    d2dexdz2_fac2_mc = metrics_savepoint.d2dexdz2_fac2_mc()
    wgtfacq_c = metrics_savepoint.wgtfacq_c_dsl()
    wgtfac_c = metrics_savepoint.wgtfac_c()
    vwind_expl_wgt = metrics_savepoint.vwind_expl_wgt()
    ddz_of_reference_exner_at_cells_on_half_levels = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()
    time_extrapolation_parameter_for_exner = metrics_savepoint.exner_exfac()
    reference_exner_at_cells_on_model_levels = metrics_savepoint.exner_ref_mc()
    inv_ddqz_z_full = metrics_savepoint.inv_ddqz_z_full()

    z_rth_pr_1_ref = sp_ref.z_rth_pr(0)
    z_rth_pr_2_ref = sp_ref.z_rth_pr(1)
    z_exner_ex_pr_ref = sp_ref.z_exner_ex_pr()
    exner_pr_ref = sp_exit.exner_pr()
    rho_ic_ref = sp_exit.rho_ic()
    z_exner_ic_ref = sp_exit.z_exner_ic()
    z_theta_v_pr_ic_ref = sp_exit.z_theta_v_pr_ic()
    theta_v_ic_ref = sp_ref.theta_v_ic()
    z_dexner_dz_c_1_ref = sp_ref.z_dexner_dz_c(0)
    z_dexner_dz_c_2_ref = sp_ref.z_dexner_dz_c(1)

    compute_cell_diagnostics_for_dycore.compute_perturbed_quantities_and_interpolation.with_backend(
        backend
    )(
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        current_rho=current_rho,
        reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
        current_theta_v=current_theta_v,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        reference_theta_at_cells_on_half_levels=reference_theta_at_cells_on_half_levels,
        wgtfacq_c=wgtfacq_c,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        pressure_buoyancy_acceleration_at_cells_on_half_levels=pressure_buoyancy_acceleration_at_cells_on_half_levels,
        time_extrapolation_parameter_for_exner=time_extrapolation_parameter_for_exner,
        current_exner=current_exner,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        inv_ddqz_z_full=inv_ddqz_z_full,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        limited_area=limited_area,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        start_cell_lateral_boundary=start_cell_lateral_boundary,
        start_cell_lateral_boundary_level_3=start_cell_lateral_boundary_level_3,
        start_cell_halo_level_2=start_cell_halo_level_2,
        end_cell_end=end_cell_end,
        end_cell_halo=end_cell_halo,
        end_cell_halo_level_2=end_cell_halo_level_2,
        horizontal_start=0,
        horizontal_end=icon_grid.num_cells,
        vertical_start=0,
        vertical_end=icon_grid.num_levels + 1,
        offset_provider={
            "Koff": dims.KDim,
        },
    )
    lb = start_cell_lateral_boundary_level_3

    assert helpers.dallclose(
        perturbed_rho_at_cells_on_model_levels.asnumpy(), z_rth_pr_1_ref.asnumpy()
    )
    assert helpers.dallclose(
        perturbed_theta_v_at_cells_on_model_levels.asnumpy(), z_rth_pr_2_ref.asnumpy()
    )
    assert helpers.dallclose(
        temporal_extrapolation_of_perturbed_exner.asnumpy(), z_exner_ex_pr_ref.asnumpy()
    )
    assert helpers.dallclose(
        perturbed_exner_at_cells_on_model_levels.asnumpy(), exner_pr_ref.asnumpy()
    )
    assert helpers.dallclose(rho_at_cells_on_half_levels.asnumpy(), rho_ic_ref.asnumpy())

    assert helpers.dallclose(
        exner_at_cells_on_half_levels.asnumpy()[:, nflatlev:],
        z_exner_ic_ref.asnumpy()[:, nflatlev:],
        rtol=1e-11,
    )

    assert helpers.dallclose(
        perturbed_theta_v_at_cells_on_half_levels.asnumpy()[lb:, :],
        z_theta_v_pr_ic_ref.asnumpy()[lb:, :],
    )
    assert helpers.dallclose(
        theta_v_at_cells_on_half_levels.asnumpy()[lb:, :], theta_v_ic_ref.asnumpy()[lb:, :]
    )

    assert helpers.dallclose(
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[lb:, nflatlev:],
        z_dexner_dz_c_1_ref.asnumpy()[lb:, nflatlev:],
        rtol=5e-9,
    )
    assert helpers.dallclose(
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels.asnumpy()[
            lb:, nflat_gradp:
        ],
        z_dexner_dz_c_2_ref.asnumpy()[lb:, nflat_gradp:],
        rtol=5e-9,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize("at_initial_timestep, istep_init, istep_exit", [(True, 2, 2)])
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
def test_interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration(
    step_date_init,
    step_date_exit,
    experiment,
    ndyn_substeps,
    icon_grid,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    at_initial_timestep,
    istep_init,
    substep_init,
    substep_exit,
    savepoint_nonhydro_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_nonhydro_exit,
    backend,
):
    sp_init = savepoint_nonhydro_init
    sp_ref = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_exit = savepoint_nonhydro_exit

    dtime = sp_init.get_metadata("dtime").get("dtime")
    current_rho = sp_init.rho_now()
    next_rho = sp_init.rho_new()

    w = sp_init.w_new()
    contravariant_correction_at_cells_on_half_levels = sp_init.w_concorr_c()
    current_theta_v = sp_init.theta_v_now()
    next_theta_v = sp_init.theta_v_new()
    perturbed_exner_at_cells_on_model_levels = sp_init.exner_pr()
    rho_at_cells_on_half_levels = sp_init.rho_ic()
    theta_v_at_cells_on_half_levels = sp_init.theta_v_ic()
    wgt_nnow_rth = sp_init.wgt_nnow_rth()
    wgt_nnew_rth = sp_init.wgt_nnew_rth()

    perturbed_theta_v_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}, backend=backend
    )
    pressure_buoyancy_acceleration_at_cells_on_half_levels = data_alloc.zero_field(
        icon_grid, dims.CellDim, dims.KDim, backend=backend
    )

    cell_domain = h_grid.domain(dims.CellDim)
    start_cell_lateral_boundary_level_3 = icon_grid.start_index(
        cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )

    end_cell_local = icon_grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    wgtfac_c = metrics_savepoint.wgtfac_c()
    reference_theta_at_cells_on_model_levels = metrics_savepoint.theta_ref_mc()
    vwind_expl_wgt = metrics_savepoint.vwind_expl_wgt()
    ddz_of_reference_exner_at_cells_on_half_levels = metrics_savepoint.d_exner_dz_ref_ic()
    ddqz_z_half = metrics_savepoint.ddqz_z_half()

    rho_ic_ref = sp_ref.rho_ic()
    z_theta_v_pr_ic_ref = sp_ref.z_theta_v_pr_ic()
    theta_v_ic_ref = sp_ref.theta_v_ic()
    z_th_ddz_exner_c_ref = sp_exit.z_th_ddz_exner_c()

    compute_cell_diagnostics_for_dycore.interpolate_rho_theta_v_to_half_levels_and_compute_pressure_buoyancy_acceleration.with_backend(
        backend
    )(
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        pressure_buoyancy_acceleration_at_cells_on_half_levels=pressure_buoyancy_acceleration_at_cells_on_half_levels,
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        current_rho=current_rho,
        next_rho=next_rho,
        current_theta_v=current_theta_v,
        next_theta_v=next_theta_v,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        wgtfac_c=wgtfac_c,
        vwind_expl_wgt=vwind_expl_wgt,
        dtime=dtime,
        wgt_nnow_rth=wgt_nnow_rth,
        wgt_nnew_rth=wgt_nnew_rth,
        horizontal_start=start_cell_lateral_boundary_level_3,
        horizontal_end=end_cell_local,
        vertical_start=1,
        vertical_end=icon_grid.num_levels,
        offset_provider={
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        rho_at_cells_on_half_levels.asnumpy()[:, :], rho_ic_ref.asnumpy()[:, :]
    )

    assert helpers.dallclose(
        theta_v_at_cells_on_half_levels.asnumpy()[:, :], theta_v_ic_ref.asnumpy()[:, :]
    )

    assert helpers.dallclose(
        perturbed_theta_v_at_cells_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        z_theta_v_pr_ic_ref.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        rtol=4e-9,
    )

    assert helpers.dallclose(
        pressure_buoyancy_acceleration_at_cells_on_half_levels.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        z_th_ddz_exner_c_ref.asnumpy()[
            start_cell_lateral_boundary_level_3:end_cell_local, 1 : icon_grid.num_levels
        ],
        rtol=5e-10,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.uses_as_offset
@pytest.mark.datatest
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
def test_compute_theta_rho_face_values_and_pressure_gradient_and_update_vn(
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
    istep_init,
    substep_init,
    substep_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    backend,
):
    sp_nh_init = savepoint_nonhydro_init
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init
    sp_stencil_exit = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit

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

    current_vn = sp_stencil_init.vn()
    next_vn = sp_nh_init.vn_new()
    tangential_wind = sp_stencil_init.vt()
    horizontal_pressure_gradient = sp_stencil_init.z_gradh_exner()
    perturbed_rho_at_cells_on_model_levels = sp_stencil_init.z_rth_pr(0)
    perturbed_theta_v_at_cells_on_model_levels = sp_stencil_init.z_rth_pr(1)
    hydrostatic_correction = data_alloc.zero_field(
        icon_grid, dims.EdgeDim, dims.KDim, backend=backend
    )
    temporal_extrapolation_of_perturbed_exner = sp_stencil_init.z_exner_ex_pr()
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        sp_stencil_init.z_dexner_dz_c(0)
    )
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        sp_stencil_init.z_dexner_dz_c(1)
    )
    theta_v = sp_stencil_init.theta_v()
    theta_v_at_cells_on_half_levels = sp_stencil_init.theta_v_ic()
    predictor_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(0)
    normal_wind_tendency_due_to_slow_physics_process = sp_stencil_init.ddt_vn_phy()
    normal_wind_iau_increments = sp_stencil_init.vn_incr()
    rho_at_edges_on_model_levels = sp_stencil_init.z_rho_e()
    theta_v_at_edges_on_model_levels = sp_stencil_init.z_theta_v_e()
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    nonhydro_params = solve_nh.NonHydrostaticParams(config)
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

    iau_wgt_dyn = config.iau_wgt_dyn
    iadv_rhotheta = config.iadv_rhotheta
    is_iau_active = config.is_iau_active
    igradp_method = config.igradp_method

    z_rho_e_ref = sp_stencil_exit.z_rho_e()
    z_theta_v_e_ref = sp_stencil_exit.z_theta_v_e()
    z_gradh_exner_ref = sp_stencil_exit.z_gradh_exner()
    vn_ref = sp_nh_exit.vn_new()

    if igradp_method == dycore_states.HorizontalPressureDiscretizationType.TAYLOR_HYDRO:
        compute_hydrostatic_correction_term.compute_hydrostatic_correction_term.with_backend(
            backend
        )(
            theta_v=theta_v,
            ikoffset=metrics_savepoint.vertoffset_gradp(),
            zdiff_gradp=metrics_savepoint.zdiff_gradp(),
            theta_v_ic=theta_v_at_cells_on_half_levels,
            inv_ddqz_z_full=metrics_savepoint.inv_ddqz_z_full(),
            inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
            z_hydro_corr=hydrostatic_correction,
            grav_o_cpd=nonhydro_params.grav_o_cpd,
            horizontal_start=start_edge_nudging_level_2,
            horizontal_end=end_edge_local,
            vertical_start=icon_grid.num_levels - 1,
            vertical_end=icon_grid.num_levels,
            offset_provider={
                "E2EC": icon_grid.get_offset_provider("E2EC"),
                "E2C": icon_grid.get_offset_provider("E2C"),
                "Koff": dims.KDim,
            },
        )
        lowest_level = icon_grid.num_levels - 1
        hydrostatic_correction_on_lowest_level = gtx.as_field(
            (dims.EdgeDim,),
            hydrostatic_correction.ndarray[:, lowest_level],
            allocator=backend,
        )
    compute_edge_diagnostics_for_dycore_and_update_vn.compute_theta_rho_face_values_and_pressure_gradient_and_update_vn.with_backend(
        backend
    )(
        rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        current_vn=current_vn,
        next_vn=next_vn,
        tangential_wind=tangential_wind,
        reference_rho_at_edges_on_model_levels=metrics_savepoint.rho_ref_me(),
        reference_theta_at_edges_on_model_levels=metrics_savepoint.theta_ref_me(),
        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increments=normal_wind_iau_increments,
        geofac_grg_x=interpolation_savepoint.geofac_grg()[0],
        geofac_grg_y=interpolation_savepoint.geofac_grg()[1],
        pos_on_tplane_e_x=interpolation_savepoint.pos_on_tplane_e_x(),
        pos_on_tplane_e_y=interpolation_savepoint.pos_on_tplane_e_y(),
        primal_normal_cell_x=primal_normal_cell_1,
        dual_normal_cell_x=dual_normal_cell_1,
        primal_normal_cell_y=primal_normal_cell_2,
        dual_normal_cell_y=dual_normal_cell_2,
        ddxn_z_full=metrics_savepoint.ddxn_z_full(),
        c_lin_e=interpolation_savepoint.c_lin_e(),
        ikoffset=metrics_savepoint.vertoffset_gradp(),
        zdiff_gradp=metrics_savepoint.zdiff_gradp(),
        ipeidx_dsl=metrics_savepoint.pg_edgeidx_dsl(),
        pg_exdist=metrics_savepoint.pg_exdist(),
        inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        cpd=constants.CPD,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        limited_area=grid_savepoint.get_metadata("limited_area").get("limited_area"),
        iadv_rhotheta=iadv_rhotheta,
        igradp_method=igradp_method,
        nflatlev=vertical_params.nflatlev,
        nflat_gradp=vertical_params.nflat_gradp,
        start_edge_halo_level_2=start_edge_halo_level_2,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        end_edge_halo=end_edge_halo,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        end_edge_end=end_edge_end,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_edges),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
            "E2EC": icon_grid.get_offset_provider("E2EC"),
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2C2EO": icon_grid.get_offset_provider("E2C2EO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(rho_at_edges_on_model_levels.asnumpy(), z_rho_e_ref.asnumpy())
    assert helpers.dallclose(theta_v_at_edges_on_model_levels.asnumpy(), z_theta_v_e_ref.asnumpy())

    assert helpers.dallclose(
        horizontal_pressure_gradient.asnumpy(),
        z_gradh_exner_ref.asnumpy(),
        atol=1e-20,
    )
    assert helpers.dallclose(
        next_vn.asnumpy()[start_edge_nudging_level_2:, :],
        vn_ref.asnumpy()[start_edge_nudging_level_2:, :],
        atol=6e-15,
    )


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init, istep_exit, substep_exit",
    [(2, 1, 2, 1)],
)
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
def test_apply_divergence_damping_and_update_vn(
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
    istep_init,
    substep_init,
    substep_exit,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init,
    savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_exit,
    backend,
):
    sp_nh_init = savepoint_nonhydro_init
    sp_nh_exit = savepoint_nonhydro_exit
    sp_stencil_init = savepoint_compute_edge_diagnostics_for_dycore_and_update_vn_init

    edge_domain = h_grid.domain(dims.EdgeDim)

    end_edge_halo_level_2 = icon_grid.end_index(edge_domain(h_grid.Zone.HALO_LEVEL_2))
    start_edge_lateral_boundary_level_7 = icon_grid.start_index(
        edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7)
    )
    start_edge_nudging_level_2 = icon_grid.start_index(edge_domain(h_grid.Zone.NUDGING_LEVEL_2))
    end_edge_local = icon_grid.end_index(edge_domain(h_grid.Zone.LOCAL))

    dwdz_at_cells_on_model_levels = sp_stencil_init.z_dwdz_dd()
    predictor_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(0)
    corrector_normal_wind_advective_tendency = sp_stencil_init.ddt_vn_apc_ntl(1)
    normal_wind_tendency_due_to_slow_physics_process = sp_stencil_init.ddt_vn_phy()
    normal_wind_iau_increments = sp_stencil_init.vn_incr()
    reduced_fourth_order_divdamp_coeff_at_nest_boundary = sp_nh_init.bdy_divdamp()
    fourth_order_divdamp_scaling_coeff = sp_nh_init.scal_divdamp()
    theta_v_at_edges_on_model_levels = sp_stencil_init.z_theta_v_e()
    horizontal_pressure_gradient = sp_stencil_init.z_gradh_exner()
    current_vn = sp_stencil_init.vn()
    next_vn = savepoint_nonhydro_init.vn_new()
    horizontal_gradient_of_normal_wind_divergence = sp_nh_init.z_graddiv_vn()
    config = utils.construct_solve_nh_config(experiment, ndyn_substeps)
    nonhydro_params = solve_nh.NonHydrostaticParams(config)

    iau_wgt_dyn = config.iau_wgt_dyn
    divdamp_order = config.divdamp_order
    second_order_divdamp_scaling_coeff = (
        sp_nh_init.divdamp_fac_o2() * grid_savepoint.mean_cell_area()
    )
    is_iau_active = config.is_iau_active

    vn_ref = sp_nh_exit.vn_new()

    compute_edge_diagnostics_for_dycore_and_update_vn.apply_divergence_damping_and_update_vn.with_backend(
        backend
    )(
        horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
        next_vn=next_vn,
        current_vn=current_vn,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increments=normal_wind_iau_increments,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        reduced_fourth_order_divdamp_coeff_at_nest_boundary=reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        fourth_order_divdamp_scaling_coeff=fourth_order_divdamp_scaling_coeff,
        second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
        horizontal_mask_for_3d_divdamp=metrics_savepoint.hmask_dd3d(),
        scaling_factor_for_3d_divdamp=metrics_savepoint.scalfac_dd3d(),
        inv_dual_edge_length=grid_savepoint.inv_dual_edge_length(),
        nudgecoeff_e=interpolation_savepoint.nudgecoeff_e(),
        geofac_grdiv=interpolation_savepoint.geofac_grdiv(),
        fourth_order_divdamp_factor=config.fourth_order_divdamp_factor,
        second_order_divdamp_factor=savepoint_nonhydro_init.divdamp_fac_o2(),
        wgt_nnow_vel=savepoint_nonhydro_init.wgt_nnow_vel(),
        wgt_nnew_vel=savepoint_nonhydro_init.wgt_nnew_vel(),
        dtime=savepoint_nonhydro_init.get_metadata("dtime").get("dtime"),
        cpd=constants.CPD,
        iau_wgt_dyn=iau_wgt_dyn,
        is_iau_active=is_iau_active,
        limited_area=grid_savepoint.get_metadata("limited_area").get("limited_area"),
        divdamp_order=divdamp_order,
        starting_vertical_index_for_3d_divdamp=nonhydro_params.starting_vertical_index_for_3d_divdamp,
        end_edge_halo_level_2=end_edge_halo_level_2,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_local=end_edge_local,
        horizontal_start=gtx.int32(0),
        horizontal_end=gtx.int32(icon_grid.num_edges),
        vertical_start=gtx.int32(0),
        vertical_end=gtx.int32(icon_grid.num_levels),
        offset_provider={
            "C2E2CO": icon_grid.get_offset_provider("C2E2CO"),
            "E2EC": icon_grid.get_offset_provider("E2EC"),
            "E2C": icon_grid.get_offset_provider("E2C"),
            "E2C2EO": icon_grid.get_offset_provider("E2C2EO"),
            "Koff": dims.KDim,
        },
    )

    assert helpers.dallclose(
        next_vn.asnumpy(),
        vn_ref.asnumpy(),
        atol=4.0e-15,
    )
