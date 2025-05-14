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
from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import states as grid_states, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import helpers, parallel_helpers

from .. import utils


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
@pytest.mark.parametrize(
    "istep_init, substep_init,istep_exit, substep_exit, at_initial_timestep",
    [(1, 1, 2, 1, True)],
)
@pytest.mark.parametrize(
    "step_date_init, step_date_exit", [("2021-06-20T12:00:10.000", "2021-06-20T12:00:10.000")]
)
@pytest.mark.mpi
def test_run_solve_nonhydro_single_step(
    istep_init,
    istep_exit,
    substep_init,
    substep_exit,
    step_date_init,
    step_date_exit,
    at_initial_timestep,
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

    exchange = definitions.create_exchange(processor_props, decomposition_info)

    config = utils.construct_solve_nh_config(experiment, ndyn=ndyn_substeps)
    nonhydro_params = nh.NonHydrostaticParams(config)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = savepoint_nonhydro_init.get_metadata("dtime").get("dtime")

    diagnostic_state_nh = utils.construct_diagnostics(savepoint_nonhydro_init, icon_grid, backend)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, icon_grid.num_levels)

    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    lprep_adv = savepoint_nonhydro_init.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=savepoint_nonhydro_init.vn_traj(),
        mass_flx_me=savepoint_nonhydro_init.mass_flx_me(),
        mass_flx_ic=savepoint_nonhydro_init.mass_flx_ic(),
        vol_flx_ic=data_alloc.zero_field(icon_grid, dims.CellDim, dims.KDim, backend=backend),
    )

    second_order_divdamp_factor = savepoint_nonhydro_init.divdamp_fac_o2()

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
    prognostic_states = utils.create_prognostic_states(savepoint_nonhydro_init)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  entering : solve_nonhydro.time_step"
    )

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
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: dycore step run ")

    expected_theta_v = savepoint_nonhydro_step_final.theta_v_new().asnumpy()
    calculated_theta_v = prognostic_states.next.theta_v.asnumpy()
    assert helpers.dallclose(
        expected_theta_v,
        calculated_theta_v,
    )
    expected_exner = savepoint_nonhydro_step_final.exner_new().asnumpy()
    calculated_exner = prognostic_states.next.exner.asnumpy()
    assert helpers.dallclose(
        expected_exner,
        calculated_exner,
    )
    assert helpers.dallclose(
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        prognostic_states.next.vn.asnumpy(),
        rtol=1e-10,
    )
    assert helpers.dallclose(
        savepoint_nonhydro_exit.w_new().asnumpy(),
        prognostic_states.next.w.asnumpy(),
        atol=8e-14,
    )
    assert helpers.dallclose(
        savepoint_nonhydro_exit.rho_new().asnumpy(),
        prognostic_states.next.rho.asnumpy(),
    )

    assert helpers.dallclose(
        savepoint_nonhydro_exit.rho_ic().asnumpy(),
        diagnostic_state_nh.rho_ic.asnumpy(),
    )

    assert helpers.dallclose(
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        diagnostic_state_nh.theta_v_at_cells_on_half_levels.asnumpy(),
    )

    assert helpers.dallclose(
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        diagnostic_state_nh.mass_fl_e.asnumpy(),
        rtol=1e-10,
    )

    assert helpers.dallclose(
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        prep_adv.mass_flx_me.asnumpy(),
        rtol=1e-10,
    )
    assert helpers.dallclose(
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        prep_adv.vn_traj.asnumpy(),
        rtol=1e-10,
    )
