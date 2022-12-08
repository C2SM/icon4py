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

import os

import numpy as np
from _pytest.fixtures import fixture

from icon4py.atm_dyn_iconam.diffusion import DiffusionConfig
from icon4py.atm_dyn_iconam.horizontal import HorizontalMeshConfig
from icon4py.atm_dyn_iconam.icon_grid import (
    IconGrid,
    MeshConfig,
    VerticalModelParams,
)
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


@fixture
def with_icon_grid():
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    sp = IconSerialDataProvider(
        "icon_diffusion_init", data_path, True
    ).from_savepoint_init(linit=True, date="2021-06-20T12:00:10.000")

    sp_meta = sp.get_metadata("nproma", "nlev", "num_vert", "num_cells", "num_edges")

    cell_starts = sp.cells_start_index()
    cell_ends = sp.cells_end_index()
    vertex_starts = sp.vertex_start_index()
    vertex_ends = sp.vertex_end_index()
    edge_starts = sp.edge_start_index()
    edge_ends = sp.edge_end_index()

    config = MeshConfig(
        HorizontalMeshConfig(
            num_vertices=sp_meta["nproma"],  # or rather "num_vert"
            num_cells=sp_meta["nproma"],  # or rather "num_cells"
            num_edges=sp_meta["nproma"],  # or rather "num_edges"
        )
    )

    c2e2c = sp.c2e2c()
    c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_starts, vertex_ends)
        .with_start_end_indices(EdgeDim, edge_starts, edge_ends)
        .with_start_end_indices(CellDim, cell_starts, cell_ends)
        .with_connectivity(c2e=sp.c2e())
        .with_connectivity(e2c=sp.e2c())
        .with_connectivity(c2e2c=c2e2c)
        .with_connectivity(e2v=sp.e2v())
        .with_connectivity(c2e2c0=c2e2c0)
        .with_connectivity(v2e=sp.v2e())
        .with_connectivity(e2v=sp.e2v())
    )
    return grid


@fixture
def with_r04b09_diffusion_config() -> DiffusionConfig:
    """
    Create DiffusionConfig.

    that uses the parameters of MCH.CH_r04b09_dsl experiment
    """
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    sp = IconSerialDataProvider(
        "icon_diffusion_init", data_path, True
    ).from_savepoint_init(linit=True, date="2021-06-20T12:00:10.000")
    nproma = sp.get_metadata("nproma")["nproma"]
    horizontalConfig = HorizontalMeshConfig(
        num_vertices=nproma, num_cells=nproma, num_edges=nproma
    )

    grid = IconGrid().with_config(MeshConfig(horizontalMesh=horizontalConfig))
    verticalParams = VerticalModelParams(
        rayleigh_damping_height=12500, vct_a=sp.vct_a()
    )
    return DiffusionConfig(grid=grid, vertical_params=verticalParams)
