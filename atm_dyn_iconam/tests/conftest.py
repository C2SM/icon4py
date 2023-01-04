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
import pytest

from icon4py.diffusion.diffusion import DiffusionConfig
from icon4py.diffusion.horizontal import HorizontalMeshConfig
from icon4py.diffusion.icon_grid import (
    IconGrid,
    MeshConfig,
    VerticalMeshConfig,
    VerticalModelParams,
)
from icon4py.common.dimension import (
    C2E2CDim,
    C2E2CODim,
    C2EDim,
    CellDim,
    E2CDim,
    E2VDim,
    EdgeDim,
    V2EDim,
    VertexDim,
)
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


data_path = os.path.join(os.path.dirname(__file__), "./ser_icondata")


@pytest.fixture
def icon_grid():

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
        ),
        VerticalMeshConfig(num_lev=sp_meta["nlev"]),
    )

    c2e2c = sp.c2e2c()
    c2e2c0 = np.column_stack((c2e2c, (np.asarray(range(c2e2c.shape[0])))))
    grid = (
        IconGrid()
        .with_config(config)
        .with_start_end_indices(VertexDim, vertex_starts, vertex_ends)
        .with_start_end_indices(EdgeDim, edge_starts, edge_ends)
        .with_start_end_indices(CellDim, cell_starts, cell_ends)
        .with_connectivities(
            {C2EDim: sp.c2e(), E2CDim: sp.e2c(), C2E2CDim: c2e2c, C2E2CODim: c2e2c0}
        )
        .with_connectivities({E2VDim: sp.e2v(), V2EDim: sp.v2e()})
    )
    return grid


@pytest.fixture
def r04b09_diffusion_config() -> DiffusionConfig:
    """
    Create DiffusionConfig.

    that uses the parameters of MCH.CH_r04b09_dsl experiment
    """
    sp = IconSerialDataProvider(
        "icon_diffusion_init", data_path, True
    ).from_savepoint_init(linit=True, date="2021-06-20T12:00:10.000")
    nproma = sp.get_metadata("nproma")["nproma"]
    num_lev = sp.get_metadata("nlev")["nlev"]
    horizontal_config = HorizontalMeshConfig(
        num_vertices=nproma, num_cells=nproma, num_edges=nproma
    )
    vertical_config = VerticalMeshConfig(num_lev=num_lev)

    grid = IconGrid().with_config(
        MeshConfig(horizontal_config=horizontal_config, vertical_config=vertical_config)
    )

    verticalParams = VerticalModelParams(
        rayleigh_damping_height=12500, vct_a=sp.vct_a()
    )

    return DiffusionConfig(
        grid=grid,
        vertical_params=verticalParams,
        diffusion_type=5,
        hdiff_w=True,
        type_t_diffu=2,
        type_vn_diffu=1,
        hdiff_efdt_ratio=24.0,
        hdiff_w_efdt_ratio=15.0,
        smag_scaling_fac=0.025,
        zdiffu_t=True,
    )
