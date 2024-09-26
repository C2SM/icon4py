# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.diffusion import diffusion as diffusion_
from icon4py.model.common import dimension as dims, settings
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.test_utils import datatest_utils, parallel_helpers

from .. import utils


@pytest.mark.mpi
@pytest.mark.parametrize("experiment", [datatest_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("linit", [True, False])
def test_parallel_diffusion(
    experiment,
    step_date_init,
    linit,
    ndyn_substeps,
    processor_props,  # fixture
    decomposition_info,
    icon_grid,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    caplog,
):
    caplog.set_level("INFO")
    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: initializing diffusion for experiment '{datatest_utils.REGIONAL_EXPERIMENT}'"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    metric_state = utils.construct_metric_state(metrics_savepoint)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    config = utils.construct_config(experiment, ndyn_substeps=ndyn_substeps)
    diffusion_params = diffusion_.DiffusionParams(config)
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    diffusion = diffusion_.Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state = utils.construct_diagnostics(diffusion_savepoint_init)
    prognostic_state = diffusion_savepoint_init.construct_prognostics()
    if linit:
        diffusion.initial_run(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
        )
    else:
        diffusion.run(
            diagnostic_state=diagnostic_state,
            prognostic_state=prognostic_state,
            dtime=dtime,
        )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion run ")

    utils.verify_diffusion_fields(
        config=config,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )


@pytest.mark.mpi
@pytest.mark.parametrize("experiment", [datatest_utils.REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize(
    "linit",
    [
        True,
    ],
)
def test_parallel_diffusion_multiple_steps(
    experiment,
    linit,
    ndyn_substeps,
    processor_props,  # fixture
    decomposition_info,
    icon_grid,
    diffusion_savepoint_init,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    lowest_layer_thickness,
    model_top_height,
    stretch_factor,
    damping_height,
    caplog,
):
    caplog.set_level("INFO")
    if settings.dace_orchestration is None:
        raise pytest.skip("This test is only executed for `--dace-orchestration=True`.")

    ######################################################################
    # Diffusion initialization
    ######################################################################
    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing diffusion for experiment '{datatest_utils.REGIONAL_EXPERIMENT}'"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(dims.CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    metric_state = utils.construct_metric_state(metrics_savepoint)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    interpolation_state = utils.construct_interpolation_state(interpolation_savepoint)
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    config = utils.construct_config(experiment, ndyn_substeps=ndyn_substeps)
    diffusion_params = diffusion_.DiffusionParams(config)
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    ######################################################################
    # DaCe NON-Orchestrated Backend
    ######################################################################
    settings.dace_orchestration = None

    diffusion = diffusion_.Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state_dace_non_orch = utils.construct_diagnostics(diffusion_savepoint_init)
    prognostic_state_dace_non_orch = diffusion_savepoint_init.construct_prognostics()
    if linit:
        for _ in range(3):
            diffusion.initial_run(
                diagnostic_state=diagnostic_state_dace_non_orch,
                prognostic_state=prognostic_state_dace_non_orch,
                dtime=dtime,
            )
    else:
        for _ in range(3):
            diffusion.run(
                diagnostic_state=diagnostic_state_dace_non_orch,
                prognostic_state=prognostic_state_dace_non_orch,
                dtime=dtime,
            )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion run ")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )

    ######################################################################
    # DaCe Orchestrated Backend
    ######################################################################
    settings.dace_orchestration = True

    diffusion = diffusion_.Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state_dace_orch = utils.construct_diagnostics(diffusion_savepoint_init)
    prognostic_state_dace_orch = diffusion_savepoint_init.construct_prognostics()
    if linit:
        for _ in range(3):
            diffusion.initial_run(
                diagnostic_state=diagnostic_state_dace_orch,
                prognostic_state=prognostic_state_dace_orch,
                dtime=dtime,
            )
    else:
        for _ in range(3):
            diffusion.run(
                diagnostic_state=diagnostic_state_dace_orch,
                prognostic_state=prognostic_state_dace_orch,
                dtime=dtime,
            )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion run ")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )

    ######################################################################
    # Verify the results
    ######################################################################
    utils.compare_dace_orchestration_multiple_steps(
        diagnostic_state_dace_non_orch, diagnostic_state_dace_orch
    )
    utils.compare_dace_orchestration_multiple_steps(
        prognostic_state_dace_non_orch, prognostic_state_dace_orch
    )
