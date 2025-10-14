# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import pytest
from gt4py.next import typing as gtx_typing

from icon4py.model.atmosphere.diffusion import diffusion as diffusion_, diffusion_states
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition, mpi_decomposition
from icon4py.model.common.grid import icon, vertical as v_grid
from icon4py.model.testing import definitions, parallel_helpers, serialbox, test_utils

from .. import utils
from ..fixtures import *  # noqa: F403


@pytest.mark.mpi
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("orchestration", [False])
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_parallel_diffusion(
    experiment: definitions.Experiment,
    step_date_init: str,
    step_date_exit: str,
    linit: bool,
    ndyn_substeps: int,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    icon_grid: icon.IconGrid,
    savepoint_diffusion_init: serialbox.IconDiffusionInitSavepoint,
    savepoint_diffusion_exit: serialbox.IconDiffusionExitSavepoint,
    grid_savepoint: serialbox.IconGridSavepoint,
    metric_state: diffusion_states.DiffusionMetricState,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    lowest_layer_thickness: ta.wpfloat,
    model_top_height: ta.wpfloat,
    stretch_factor: ta.wpfloat,
    damping_height: ta.wpfloat,
    caplog: Any,
    backend: gtx_typing.Backend,
    orchestration: bool,
) -> None:
    if orchestration and not test_utils.is_dace(backend):
        raise pytest.skip("This test is only executed for `dace` backends.")
    caplog.set_level("INFO")
    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: initializing diffusion for experiment '{definitions.Experiments.MCH_CH_R04B09}'"
    )
    print(
        f"local cells = {decomposition_info.global_index(dims.CellDim, decomposition.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, decomposition.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, decomposition.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    config = definitions.construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )

    diffusion_params = diffusion_.DiffusionParams(config)
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    diffusion = diffusion_.Diffusion(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config,
            grid_savepoint.vct_a(),
            grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        exchange=exchange,
        backend=backend,
        orchestration=orchestration,
    )

    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")

    diagnostic_state = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )

    prognostic_state = savepoint_diffusion_init.construct_prognostics()
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
        diffusion_savepoint=savepoint_diffusion_exit,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )


@pytest.mark.mpi
@pytest.mark.parametrize(
    "experiment, step_date_init, step_date_exit",
    [
        (
            definitions.Experiments.MCH_CH_R04B09,
            "2021-06-20T12:00:10.000",
            "2021-06-20T12:00:10.000",
        ),
        (definitions.Experiments.EXCLAIM_APE, "2000-01-01T00:00:02.000", "2000-01-01T00:00:02.000"),
    ],
)
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_parallel_diffusion_multiple_steps(
    experiment: definitions.Experiment,
    step_date_init: str,
    step_date_exit: str,
    linit: bool,
    ndyn_substeps: int,
    processor_props: decomposition.ProcessProperties,
    decomposition_info: decomposition.DecompositionInfo,
    icon_grid: icon.IconGrid,
    savepoint_diffusion_init: serialbox.IconDiffusionInitSavepoint,
    grid_savepoint: serialbox.IconGridSavepoint,
    metric_state: diffusion_states.DiffusionMetricState,
    interpolation_state: diffusion_states.DiffusionInterpolationState,
    lowest_layer_thickness: ta.wpfloat,
    model_top_height: ta.wpfloat,
    stretch_factor: ta.wpfloat,
    damping_height: ta.wpfloat,
    caplog: Any,
    backend: gtx_typing.Backend | None,
):
    if not test_utils.is_dace(backend):
        raise pytest.skip("This test is only executed for `dace backends.")
    ######################################################################
    # Diffusion initialization
    ######################################################################
    caplog.set_level("INFO")
    parallel_helpers.check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: initializing diffusion for experiment '{definitions.Experiments.MCH_CH_R04B09}'"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(dims.CellDim, decomposition.DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(dims.EdgeDim, decomposition.DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(dims.VertexDim, decomposition.DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells} Cells, {icon_grid.num_edges} Edges, {icon_grid.num_vertices} Vertices"
    )
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()

    vertical_config = v_grid.VerticalGridConfig(
        icon_grid.num_levels,
        lowest_layer_thickness=lowest_layer_thickness,
        model_top_height=model_top_height,
        stretch_factor=stretch_factor,
        rayleigh_damping_height=damping_height,
    )
    config = definitions.construct_diffusion_config(experiment, ndyn_substeps=ndyn_substeps)
    diffusion_params = diffusion_.DiffusionParams(config)
    dtime = savepoint_diffusion_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = decomposition.create_exchange(processor_props, decomposition_info)

    ######################################################################
    # DaCe NON-Orchestrated Backend
    ######################################################################

    diffusion = diffusion_.Diffusion(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config,
            grid_savepoint.vct_a(),
            grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        backend=backend,
        exchange=exchange,
        orchestration=False,
    )

    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")

    diagnostic_state_dace_non_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )

    prognostic_state_dace_non_orch = savepoint_diffusion_init.construct_prognostics()
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

    exchange = decomposition.create_exchange(processor_props, decomposition_info)
    diffusion = diffusion_.Diffusion(
        grid=icon_grid,
        config=config,
        params=diffusion_params,
        vertical_grid=v_grid.VerticalGrid(
            vertical_config,
            grid_savepoint.vct_a(),
            grid_savepoint.vct_b(),
        ),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
        exchange=exchange,
        backend=backend,
        orchestration=True,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")

    diagnostic_state_dace_orch = diffusion_states.DiffusionDiagnosticState(
        hdef_ic=savepoint_diffusion_init.hdef_ic(),
        div_ic=savepoint_diffusion_init.div_ic(),
        dwdx=savepoint_diffusion_init.dwdx(),
        dwdy=savepoint_diffusion_init.dwdy(),
    )

    prognostic_state_dace_orch = savepoint_diffusion_init.construct_prognostics()
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
