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

from icon4py.atm_dyn_iconam.horizontal import (
    HorizontalMarkerIndex,
    HorizontalMeshConfig,
)
from icon4py.atm_dyn_iconam.icon_grid import IconGrid, MeshConfig
from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.testutils.serialbox_utils import IconSerialDataProvider


@fixture
def with_grid():
    data_path = os.path.join(os.path.dirname(__file__), "ser_icondata")
    sp = IconSerialDataProvider("icon_diffusion_init", data_path).from_savepoint(
        linit=True, date="2021-06-20T12:00:10.000"
    )
    nproma, nlev, num_v, num_c, num_e = sp.get_metadata(
        "nproma", "nlev", "num_vert", "num_cells", "num_edges"
    )
    cell_starts = sp.cells_start_index()
    cell_ends = sp.cells_end_index()
    vertex_starts = sp.vertex_start_index()
    vertex_ends = sp.vertex_end_index()
    edge_starts = sp.edge_start_index()
    edge_ends = sp.edge_end_index()

    config = MeshConfig(
        HorizontalMeshConfig(num_vertices=num_v, num_cells=num_c, num_edges=num_e)
    )
    c2e2c = np.squeeze(sp.c2e2c(), axis=1)
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
    )
    return grid


def test_horizontal_grid_cell_indices(with_grid):
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.halo(CellDim) - 1,
        HorizontalMarkerIndex.halo(CellDim) - 1,
    ) == (
        20897,
        20896,
    )  # halo +1
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.halo(CellDim),
        HorizontalMarkerIndex.halo(CellDim),
    ) == (
        0,
        20896,
    )  # halo in icon is (1,20896) why
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.interior(CellDim),
        HorizontalMarkerIndex.interior(CellDim),
    ) == (
        4105,
        20896,
    )  # interior
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.interior(CellDim) + 1,
        HorizontalMarkerIndex._INTERIOR_CELLS + 1,
    ) == (
        1,
        850,
    )  # lb+1
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.local_boundary(CellDim) + 1,
        HorizontalMarkerIndex.local_boundary(CellDim) + 1,
    ) == (851, 1688)
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.local_boundary(CellDim) + 2,
        HorizontalMarkerIndex.local_boundary(CellDim) + 2,
    ) == (
        1689,
        2511,
    )  # lb+2
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
    ) == (
        2512,
        3316,
    )  # lb+3
    assert with_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.nudging(CellDim),
        HorizontalMarkerIndex.nudging(CellDim),
    ) == (
        3317,
        4104,
    )  # nudging


def test_horizontal_edge_indices(with_grid):
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.interior(EdgeDim),
        HorizontalMarkerIndex.interior(EdgeDim),
    ) == (6177, 31558)
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.halo(EdgeDim) - 2,
        HorizontalMarkerIndex.halo(EdgeDim) - 2,
    ) == (31559, 31558)
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.halo(EdgeDim) - 1,
        HorizontalMarkerIndex.halo(EdgeDim) - 1,
    ) == (31559, 31558)
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.halo(EdgeDim),
        HorizontalMarkerIndex.halo(EdgeDim),
    ) == (
        0,
        31558,
    )  # halo in icon is  (1, 31558)
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.nudging(EdgeDim) + 1,
        HorizontalMarkerIndex.nudging(EdgeDim) + 1,
    ) == (
        5388,
        6176,
    )  # nudging +1
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.nudging(EdgeDim),
        HorizontalMarkerIndex.nudging(EdgeDim),
    ) == (
        4990,
        5387,
    )  # nudging
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 7,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 7,
    ) == (
        4185,
        4989,
    )  # lb +7
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 6,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 6,
    ) == (
        3778,
        4184,
    )  # lb +6
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 5,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 5,
    ) == (
        2955,
        3777,
    )  # lb +5
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 4,
    ) == (
        2539,
        2954,
    )  # lb +4
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 3,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 3,
    ) == (
        1701,
        2538,
    )  # lb +3
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 2,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 2,
    ) == (
        1279,
        1700,
    )  # lb +2
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 1,
        HorizontalMarkerIndex.local_boundary(EdgeDim) + 1,
    ) == (
        429,
        1278,
    )  # lb +1
    assert with_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local_boundary(EdgeDim),
        HorizontalMarkerIndex.local_boundary(EdgeDim),
    ) == (
        1,
        428,
    )  # lb +0


def test_horizontal_vertex_indices(with_grid):
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.end(VertexDim),
        HorizontalMarkerIndex.end(VertexDim),
    ) == (10664, 10663)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.halo(VertexDim),
        HorizontalMarkerIndex.halo(VertexDim),
    ) == (0, 10663)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.halo(VertexDim) - 1,
        HorizontalMarkerIndex.halo(VertexDim) - 1,
    ) == (10664, 10663)

    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local_boundary(VertexDim),
        HorizontalMarkerIndex.local_boundary(VertexDim),
    ) == (1, 428)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 1,
    ) == (429, 850)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 2,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 2,
    ) == (851, 1266)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 3,
    ) == (1267, 1673)
    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 4,
        HorizontalMarkerIndex.local_boundary(VertexDim) + 4,
    ) == (1674, 2071)

    assert with_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.interior(VertexDim),
        HorizontalMarkerIndex.interior(VertexDim),
    ) == (2072, 10663)
