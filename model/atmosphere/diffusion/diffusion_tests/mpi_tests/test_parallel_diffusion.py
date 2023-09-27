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


from icon4py.model.atmosphere.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.model.common.decomposition.decomposed import DecompositionInfo, create_exchange
from icon4py.model.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.model.common.grid.vertical import VerticalModelParams
from icon4py.model.common.test_utils.parallel_helpers import check_comm_size

import pytest

from ..utils import verify_diffusion_fields


@pytest.mark.mpi
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("linit", [True, False])
@pytest.mark.parametrize("processor_props", [True], indirect=True)
def test_parallel_diffusion(
    r04b09_diffusion_config,
    step_date_init,
    linit,
    ndyn_substeps,
    processor_props,
    decomposition_info,
    icon_grid,
    diffusion_savepoint_init,
    diffusion_savepoint_exit,
    grid_savepoint,
    metrics_savepoint,
    interpolation_savepoint,
    damping_height,
):
    check_comm_size(processor_props)
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: inializing diffusion for experiment 'mch_ch_r04_b09_dsl"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: decomposition info : klevels = {decomposition_info.klevels}, "
        f"local cells = {decomposition_info.global_index(CellDim, DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomposition_info.global_index(EdgeDim, DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomposition_info.global_index(VertexDim, DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  GHEX context setup: from {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )

    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}: using local grid with {icon_grid.num_cells()} Cells, {icon_grid.num_edges()} Edges, {icon_grid.num_vertices()} Vertices"
    )
    metric_state = metrics_savepoint.construct_metric_state_for_diffusion()
    cell_geometry = grid_savepoint.construct_cell_geometry()
    edge_geometry = grid_savepoint.construct_edge_geometry()
    interpolation_state = interpolation_savepoint.construct_interpolation_state_for_diffusion()

    diffusion_params = DiffusionParams(r04b09_diffusion_config)
    dtime = diffusion_savepoint_init.get_metadata("dtime").get("dtime")
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  setup: using {processor_props.comm_name} with {processor_props.comm_size} nodes"
    )
    exchange = create_exchange(processor_props, decomposition_info)

    diffusion = Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=r04b09_diffusion_config,
        params=diffusion_params,
        vertical_params=VerticalModelParams(grid_savepoint.vct_a(), damping_height),
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={processor_props.rank}/{processor_props.comm_size}: diffusion initialized ")
    diagnostic_state = diffusion_savepoint_init.construct_diagnostics_for_diffusion()
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
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    print(
        f"rank={processor_props.rank}/{processor_props.comm_size}:  running diffusion step - using {processor_props.comm_name} with {processor_props.comm_size} nodes - DONE"
    )
