# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as nh
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import icon, states as grid_states, vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, parallel_helpers, serialbox, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


@pytest.mark.parametrize("processor_props", [True], indirect=True)
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment, istep_init, step_date_init, substep_init, istep_exit, step_date_exit, substep_exit",
    [
        (
            test_defs.Experiments.MCH_CH_R04B09,
            1,
            "2021-06-20T12:00:10.000",
            1,
            2,
            "2021-06-20T12:00:10.000",
            1,
        ),
        (
            test_defs.Experiments.GAUSS3D,
            1,
            "2001-01-01T00:00:04.000",
            1,
            2,
            "2001-01-01T00:00:04.000",
            1,
        ),
        (
            test_defs.Experiments.EXCLAIM_APE,
            1,
            "2000-01-01T00:00:02.000",
            1,
            2,
            "2000-01-01T00:00:02.000",
            1,
        ),
    ],
)
@pytest.mark.mpi
def test_run_solve_nonhydro_single_step(
    istep_init: int,
    istep_exit: int,
    step_date_init: str,
    step_date_exit: str,
    substep_init: int,
    experiment: test_defs.Experiment,
    ndyn_substeps: int,
    icon_grid: icon.IconGrid,
    savepoint_nonhydro_init: serialbox.IconNonHydroInitSavepoint,
    lowest_layer_thickness: ta.wpfloat,
    model_top_height: ta.wpfloat,
    stretch_factor: ta.wpfloat,
    damping_height: ta.wpfloat,
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    savepoint_nonhydro_exit: serialbox.IconNonHydroExitSavepoint,
    savepoint_nonhydro_step_final: serialbox.IconNonHydroFinalSavepoint,
    processor_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,  # : F811 fixture
    backend: gtx_typing.Backend | None,
) -> None:
    if test_utils.is_embedded(backend):
        # https://github.com/GridTools/gt4py/issues/1583
        pytest.xfail("ValueError: axes don't match array")

    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing dycore for experiment 'mch_ch_r04_b09_dsl"
    )
    print(
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

    config = test_defs.construct_nonhydrostatic_config(experiment)
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
    lprep_adv = savepoint_nonhydro_init.get_metadata("prep_adv").get("prep_adv")
    prep_adv = dycore_states.PrepAdvection(
        vn_traj=savepoint_nonhydro_init.vn_traj(),
        mass_flx_me=savepoint_nonhydro_init.mass_flx_me(),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=savepoint_nonhydro_init.mass_flx_ic(),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, allocator=backend
        ),
    )

    diagnostic_state_nh = utils.construct_diagnostics(savepoint_nonhydro_init, icon_grid, backend)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)
    second_order_divdamp_factor = savepoint_nonhydro_init.divdamp_fac_o2()
    at_initial_timestep = True
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_states = utils.create_prognostic_states(savepoint_nonhydro_init)

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
        at_initial_timestep=at_initial_timestep,
        lprep_adv=lprep_adv,
        at_first_substep=(substep_init == 1),
        at_last_substep=(substep_init == ndyn_substeps),
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: dycore step run ")

    expected_theta_v = savepoint_nonhydro_step_final.theta_v_new().asnumpy()
    calculated_theta_v = prognostic_states.next.theta_v.asnumpy()
    assert test_utils.dallclose(
        expected_theta_v,
        calculated_theta_v,
    )
    expected_exner = savepoint_nonhydro_step_final.exner_new().asnumpy()
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
        diagnostic_state_nh.rho_at_cells_on_half_levels.asnumpy(),
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
