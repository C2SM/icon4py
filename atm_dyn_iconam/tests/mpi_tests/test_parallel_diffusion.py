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

from atm_dyn_iconam.tests.mpi_tests.common import path, props
from atm_dyn_iconam.tests.test_diffusion import _verify_diffusion_fields
from atm_dyn_iconam.tests.test_utils.serialbox_utils import IconSerialDataProvider
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.decomposition.decomposed import DecompositionInfo, create_exchange
from icon4py.diffusion.diffusion import Diffusion, DiffusionParams
from icon4py.driver.io_utils import (
    read_decomp_info,
    read_geometry_fields,
    read_icon_grid,
    read_static_fields,
)


@pytest.mark.mpi
@pytest.mark.parametrize("ndyn_substeps", [2])
@pytest.mark.parametrize("linit", [True, False])
@pytest.mark.skipif(
    props.comm_size not in (1, 2, 4),
    reason="input files only available for 1 or 2 nodes",
)
def test_parallel_diffusion(
    r04b09_diffusion_config, step_date_init, linit, ndyn_substeps
):

    print(
        f"rank={props.rank}/{props.comm_size}: inializing diffusion for experiment 'mch_ch_r04_b09_dsl"
    )
    decomp_info = read_decomp_info(
        path,
        props,
    )
    print(
        f"rank={props.rank}/{props.comm_size}: decomposition info : klevels = {decomp_info.klevels}, "
        f"local cells = {decomp_info.global_index(CellDim, DecompositionInfo.EntryType.ALL).shape} "
        f"local edges = {decomp_info.global_index(EdgeDim, DecompositionInfo.EntryType.ALL).shape} "
        f"local vertices = {decomp_info.global_index(VertexDim, DecompositionInfo.EntryType.ALL).shape}"
    )
    print(
        f"rank={props.rank}/{props.comm_size}:  GHEX context setup: from {props.comm_name} with {props.comm_size} nodes"
    )

    icon_grid = read_icon_grid(path, rank=props.rank)
    print(
        f"rank={props.rank}: using local grid with {icon_grid.num_cells()} Cells, {icon_grid.num_edges()} Edges, {icon_grid.num_vertices()} Vertices"
    )
    diffusion_params = DiffusionParams(r04b09_diffusion_config)

    diffusion_initial_data = IconSerialDataProvider(
        "icon_pydycore", str(path), True, mpi_rank=props.rank
    ).from_savepoint_diffusion_init(linit=linit, date=step_date_init)
    (edge_geometry, cell_geometry, vertical_geometry) = read_geometry_fields(
        path, rank=props.rank
    )
    (metric_state, interpolation_state) = read_static_fields(path, rank=props.rank)

    dtime = diffusion_initial_data.get_metadata("dtime").get("dtime")
    print(
        f"rank={props.rank}/{props.comm_size}:  setup: using {props.comm_name} with {props.comm_size} nodes"
    )
    exchange = create_exchange(props, decomp_info)

    diffusion = Diffusion(exchange)

    diffusion.init(
        grid=icon_grid,
        config=r04b09_diffusion_config,
        params=diffusion_params,
        vertical_params=vertical_geometry,
        metric_state=metric_state,
        interpolation_state=interpolation_state,
        edge_params=edge_geometry,
        cell_params=cell_geometry,
    )
    print(f"rank={props.rank}/{props.comm_size}: diffusion initialized ")
    diagnostic_state = diffusion_initial_data.construct_diagnostics_for_diffusion()
    prognostic_state = diffusion_initial_data.construct_prognostics()
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
    print(f"rank={props.rank}/{props.comm_size}: diffusion run ")

    diffusion_savepoint_exit = IconSerialDataProvider(
        "icon_pydycore", str(path), True, mpi_rank=props.rank
    ).from_savepoint_diffusion_exit(linit=linit, date=step_date_init)
    _verify_diffusion_fields(
        diagnostic_state=diagnostic_state,
        prognostic_state=prognostic_state,
        diffusion_savepoint=diffusion_savepoint_exit,
    )
    print(
        f"rank={props.rank}/{props.comm_size}:  running diffusion step - using {props.comm_name} with {props.comm_size} nodes - DONE"
    )

