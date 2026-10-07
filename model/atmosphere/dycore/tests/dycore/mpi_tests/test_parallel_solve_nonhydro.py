# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import logging

import numpy as np
import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as nh
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions, mpi_decomposition
from icon4py.model.common.grid import (
    horizontal as h_grid,
    icon,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, parallel_helpers, serialbox, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


_log = logging.getLogger(__name__)


@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, istep_init, step_date_init, substep_init, istep_exit, step_date_exit, substep_exit",
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
    ],
)
@pytest.mark.mpi
def test_run_solve_nonhydro_single_step(  # noqa: PLR0917 [too-many-positional-arguments]
    istep_init: int,
    istep_exit: int,
    step_date_init: str,
    step_date_exit: str,
    substep_init: int,
    experiment: test_defs.Experiment,
    icon_grid: icon.IconGrid,
    savepoint_nonhydro_init: serialbox.IconNonHydroInitSavepoint,
    is_iau_active: bool,
    iau_wgt_dyn: ta.wpfloat,
    grid_savepoint: serialbox.IconGridSavepoint,
    metrics_savepoint: serialbox.MetricSavepoint,
    interpolation_savepoint: serialbox.InterpolationSavepoint,
    savepoint_nonhydro_exit: serialbox.IconNonHydroExitSavepoint,
    savepoint_nonhydro_step_final: serialbox.IconNonHydroFinalSavepoint,
    process_props: definitions.ProcessProperties,
    decomposition_info: definitions.DecompositionInfo,  # : F811 fixture
    backend: gtx_typing.Backend | None,
) -> None:
    if test_utils.is_embedded(backend):
        # https://github.com/GridTools/gt4py/issues/1583
        pytest.xfail("ValueError: axes don't match array")

    parallel_helpers.check_comm_size(process_props)
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: inializing dycore for experiment 'mch_ch_r04_b09_dsl"
    )
    _log.info(
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    owned_cells = decomposition_info.owner_mask(dims.CellDim)
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  GHEX context setup: from {process_props.comm_name} with {process_props.comm_size} nodes"
    )
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: number of halo edges {np.count_nonzero(np.invert(decomposition_info.owner_mask(dims.EdgeDim)))}"
    )
    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}: number of halo cells {np.count_nonzero(np.invert(owned_cells))}"
    )

    assert experiment.config.nonhydrostatic is not None

    nonhydro_params = nh.NonHydrostaticParams(experiment.config.nonhydrostatic)
    vertical_config = experiment.config.vertical_grid
    vertical_params = utils.create_vertical_params(vertical_config, grid_savepoint)
    dtime = savepoint_nonhydro_init.dtime()
    prepare_fluxes_for_advection = savepoint_nonhydro_init.get_metadata("prep_adv").get("prep_adv")
    prep_adv = utils.construct_prep_advection(savepoint_nonhydro_init, icon_grid, backend)
    forcing = utils.construct_forcing(savepoint_nonhydro_init, icon_grid, backend)
    dycore_diagnostics = utils.construct_dycore_diagnostics(savepoint_nonhydro_init)

    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    metric_state_nonhydro = utils.construct_metric_state(metrics_savepoint, grid_savepoint)
    second_order_divdamp_factor = savepoint_nonhydro_init.divdamp_fac_o2()
    at_initial_timestep = True
    cell_geometry: grid_states.CellParams = grid_savepoint.construct_cell_geometry()
    edge_geometry: grid_states.EdgeParams = grid_savepoint.construct_edge_geometry()

    prognostic_states = utils.create_prognostic_states(savepoint_nonhydro_init)

    exchange = definitions.create_exchange(process_props, decomposition_info)

    solve_nonhydro = nh.SolveNonhydro(
        grid=icon_grid,
        config=experiment.config.nonhydrostatic,
        params=nonhydro_params,
        metric_state_nonhydro=metric_state_nonhydro,
        interpolation_state=interpolation_state,
        vertical_params=vertical_params,
        edge_geometry=edge_geometry,
        cell_geometry=cell_geometry,
        owner_mask=grid_savepoint.c_owner_mask(),
        backend=backend,
        exchange=exchange,
        max_nudging_coefficient=experiment.config.interpolation.max_nudging_coefficient,
    )

    _log.info(
        f"rank={process_props.rank}/{process_props.comm_size}:  entering : solve_nonhydro.time_step"
    )

    solve_nonhydro.diagnostics = utils.construct_diagnostics(savepoint_nonhydro_init)
    solve_nonhydro.run(
        utils.dycore_inputs(
            prognostic_states.current,
            forcing,
            second_order_divdamp_factor=second_order_divdamp_factor,
            dtime=dtime,
            ndyn_substeps_var=experiment.config.driver.ndyn_substeps,
            at_initial_timestep=at_initial_timestep,
            prepare_fluxes_for_advection=prepare_fluxes_for_advection,
            at_first_substep=(substep_init == 1),
            at_last_substep=(substep_init == experiment.config.driver.ndyn_substeps),
            is_iau_active=is_iau_active,
            iau_wgt_dyn=iau_wgt_dyn,
        ),
        utils.dycore_output(prognostic_states.next, prep_adv, dycore_diagnostics),
    )
    _log.info(f"rank={process_props.rank}/{process_props.comm_size}: dycore step run ")

    expected_theta_v = savepoint_nonhydro_step_final.theta_v_new().asnumpy()
    calculated_theta_v = prognostic_states.next.theta_v.data.asnumpy()
    assert test_utils.dallclose(
        expected_theta_v,
        calculated_theta_v,
    )
    expected_exner = savepoint_nonhydro_step_final.exner_new().asnumpy()
    calculated_exner = prognostic_states.next.exner.data.asnumpy()
    assert test_utils.dallclose(
        expected_exner,
        calculated_exner,
    )
    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.vn_new().asnumpy(),
        prognostic_states.next.vn.data.asnumpy(),
        atol=1e-14,
        rtol=1e-10,
    )
    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.w_new().asnumpy(),
        prognostic_states.next.w.data.asnumpy(),
        atol=1e-14,
    )
    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.rho_new().asnumpy(),
        prognostic_states.next.rho.data.asnumpy(),
    )

    # `rho_ic` is only computed on locally owned cells, the reference contains ICON's halo values.
    end_cell_local = icon_grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LOCAL))
    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.rho_ic().asnumpy()[:end_cell_local, :],
        solve_nonhydro.diagnostics.rho_at_cells_on_half_levels.data.asnumpy()[:end_cell_local, :],
    )

    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.theta_v_ic().asnumpy(),
        solve_nonhydro.diagnostics.theta_v_at_cells_on_half_levels.data.asnumpy(),
    )

    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.mass_fl_e().asnumpy(),
        solve_nonhydro.diagnostics.mass_flux_at_edges_on_model_levels.data.asnumpy(),
        rtol=1e-10,
    )

    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.mass_flx_me().asnumpy(),
        prep_adv.mass_flx_me.data.asnumpy(),
        rtol=1e-10,
    )
    test_utils.assert_dallclose(
        savepoint_nonhydro_exit.vn_traj().asnumpy(),
        prep_adv.vn_traj.data.asnumpy(),
        rtol=1e-10,
    )
