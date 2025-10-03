# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as nh
from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import parallel_helpers, test_utils

from .. import utils


@pytest.mark.skip("FIXME: Need updated test data yet")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, jstep_init, step_date_init,istep_exit, jstep_exit, step_date_exit",
    [(1, 0, "2021-06-20T12:00:10.000", 2, 0, "2021-06-20T12:00:10.000")],
)
@pytest.mark.mpi
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
    savepoint_velocity_init,
    metrics_savepoint,
    interpolation_savepoint,
    savepoint_nonhydro_exit,
    savepoint_nonhydro_step_final,
    processor_props,  # : F811 fixture
    decomposition_info,  # : F811 fixture
    backend,
):
    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing dycore for experiment 'mch_ch_r04_b09_dsl"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels} "
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    owned_cells = decomposition_info.owner_mask(dims.CellDim)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo edges {np.count_nonzero(np.invert( decomposition_info.owner_mask(dims.EdgeDim)))}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )

    config = definitions.construct_nonhydrostatic_config(experiment)
    sp = savepoint_nonhydro_init
    sp_step_exit = savepoint_nonhydro_step_final
    nonhydro_params = nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
        _min_index_flat_horizontal_grad_pressure=grid_savepoint.nflat_gradp(),
    )
    sp_v = savepoint_velocity_init
    dtime = sp_v.get_metadata("dtime").get("dtime")
    lprep_adv = sp_v.get_metadata("prep_adv").get("prep_adv")
    # clean_mflx = sp_v.get_metadata("clean_mflx").get("clean_mflx")  # noqa: ERA001 [commented-out-code]
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=sp.vn_traj(),
        mass_flx_me=sp.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=sp.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),
    )

    recompute = sp_v.get_metadata("recompute").get("recompute")
    # linit = sp_v.get_metadata("linit").get("linit")  # noqa: ERA001 [commented-out-code]

    diagnostic_state_nh = dycore_states.DiagnosticStateNonHydro(
        max_vertical_cfl=0.0,
        theta_v_at_cells_on_half_levels=sp.theta_v_ic(),
        perturbed_exner_at_cells_on_model_levels=sp.exner_pr(),
        rho_at_cells_on_half_levels=sp.rho_ic(),
        exner_tendency_due_to_slow_physics=sp.ddt_exner_phy(),
        grf_tend_rho=sp.grf_tend_rho(),
        grf_tend_thv=sp.grf_tend_thv(),
        grf_tend_w=sp.grf_tend_w(),
        mass_flux_at_edges_on_model_levels=sp.mass_fl_e(),
        normal_wind_tendency_due_to_slow_physics_process=sp.ddt_vn_phy(),
        grf_tend_vn=sp.grf_tend_vn(),
        normal_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp_v.ddt_vn_apc_pc(1), sp_v.ddt_vn_apc_pc(2)
        ),
        vertical_wind_advective_tendency=common_utils.PredictorCorrectorPair(
            sp_v.ddt_w_adv_pc(1), sp_v.ddt_w_adv_pc(2)
        ),
        tangential_wind=sp_v.vt(),
        vn_on_half_levels=sp_v.vn_ie(),
        contravariant_correction_at_cells_on_half_levels=sp_v.w_concorr_c(),
        rho_iau_increment=None,  # sp.rho_incr(),
        normal_wind_iau_increment=None,  # sp.vn_incr(),
        exner_iau_increment=None,  # sp.exner_incr(),
        exner_dynamical_increment=sp.exner_dyn_incr(),
    )
    second_order_divdamp_factor = sp.divdamp_fac_o2()
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_states = utils.create_prognostic_states(sp)

    exchange = definitions.create_exchange(processor_props, decomposition_info)

    solve_nonhydro = nh.SolveNonhydro(
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
        exchange=exchange,
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  entering : solve_nonhydro.time_step"
    )

    solve_nonhydro.time_step(
        diagnostic_state_nh=diagnostic_state_nh,
        prognostic_states=prognostic_states,
        prep_adv=prep_adv,
        second_order_divdamp_factor=second_order_divdamp_factor,
        dtime=dtime,
        ndyn_substeps_var=ndyn_substeps,
        at_initial_timestep=recompute,
        lprep_adv=lprep_adv,
        at_first_substep=jstep_init == 0,
        at_last_substep=jstep_init == (ndyn_substeps - 1),
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: dycore step run ")

    expected_theta_v = sp_step_exit.theta_v_new().asnumpy()
    calculated_theta_v = prognostic_states.next.theta_v.asnumpy()
    assert test_utils.dallclose(
        expected_theta_v,
        calculated_theta_v,
    )
    expected_exner = sp_step_exit.exner_new().asnumpy()
    calculated_exner = prognostic_states.next.exner.asnumpy()
    assert test_utils.dallclose(
        expected_exner,
        calculated_exner,
    )
    assert test_utils.dallclose(
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        prognostic_states.next.vn.asnumpy(),
        rtol=1e-10,
    )
    assert test_utils.dallclose(
        savepoint_nonhydro_exit.w_new().asnumpy(),
        prognostic_states.next.w.asnumpy(),
        atol=8e-14,
    )
    assert test_utils.dallclose(
        savepoint_nonhydro_exit.rho_new().asnumpy(),
        prognostic_states.next.rho.asnumpy(),
    )

    assert test_utils.dallclose(
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
        diagnostic_state_nh.rho_ic.asnumpy(),
    )

    assert test_utils.dallclose(
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy(),
    )

    assert test_utils.dallclose(
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        diagnostic_state_nh.mass_flux_at_edges_on_model_levels.asnumpy(),
        rtol=1e-10,
    )

    assert test_utils.dallclose(
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        prep_adv.mass_flx_me.asnumpy(),
        rtol=1e-10,
    )
    assert test_utils.dallclose(
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        prep_adv.vn_traj.asnumpy(),
        rtol=1e-10,
    )
