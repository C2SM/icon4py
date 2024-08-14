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


import pytest

from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.common import settings
from icon4py.model.common.decomposition import definitions
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.model.common.grid.vertical import VerticalGridConfig, VerticalGridParams
from icon4py.model.common.orchestration.decorator import dace_orchestration
from icon4py.model.common.test_utils.datatest_utils import REGIONAL_EXPERIMENT
from icon4py.model.common.test_utils.parallel_helpers import (  # noqa: F401  # import fixtures from test_utils package
    check_comm_size,
    processor_props,
)

from ..utils import (
    compare_dace_orchestration_multiple_steps,
    construct_config,
    construct_diagnostics,
    construct_interpolation_state,
    construct_metric_state,
    verify_diffusion_fields,
)


@pytest.mark.mpi
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("linit", [True, False])
def test_parallel_diffusion(
    experiment,
    step_date_init,
    linit,
    ndyn_substeps,
    processor_props,  # noqa: F811  # fixture
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
):
    check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing diffusion for experiment '{REGIONAL_EXPERIMENT}'"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    metric_state = construct_metric_state(metrics_savepoint)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    vertical_config = VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    diffusion_params = DiffusionParams(config)
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    diffusion = Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=VerticalGridParams(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state = construct_diagnostics(diffusion_savepoint_init)
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

    verify_diffusion_fields(
        config=config,
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )


@pytest.mark.mpi
@pytest.mark.parametrize("experiment", [REGIONAL_EXPERIMENT])
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("linit", [True, False])
def test_parallel_diffusion_multiple_steps(
    experiment,
    step_date_init,
    linit,
    ndyn_substeps,
    processor_props,  # noqa: F811  # fixture
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
):
    if not dace_orchestration():
        raise pytest.skip("This test is only executed for `--dace-orchestration=True`.")

    ######################################################################
    # Diffusion initialization
    ######################################################################
    check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing diffusion for experiment '{REGIONAL_EXPERIMENT}'"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(CellDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, definitions.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, definitions.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    metric_state = construct_metric_state(metrics_savepoint)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    interpolation_state = construct_interpolation_state(interpolation_savepoint)
    vertical_config = VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    config = construct_config(experiment, ndyn_substeps=ndyn_substeps)
    diffusion_params = DiffusionParams(config)
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = definitions.create_exchange(processor_props, decomposition_info)

    ######################################################################
    # DaCe NON-Orchestrated Backend
    ######################################################################
    settings.dace_orchestration = None

    diffusion = Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=VerticalGridParams(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state_dace_non_orch = construct_diagnostics(diffusion_savepoint_init)
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

    diffusion = Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_params=VerticalGridParams(
            vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state_dace_orch = construct_diagnostics(diffusion_savepoint_init)
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
    compare_dace_orchestration_multiple_steps(
        diagnostic_state_dace_non_orch, diagnostic_state_dace_orch
    )
    compare_dace_orchestration_multiple_steps(
        prognostic_state_dace_non_orch, prognostic_state_dace_orch
    )
