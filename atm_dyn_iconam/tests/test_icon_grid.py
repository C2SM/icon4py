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

from icon4py.common.dimension import CellDim, EdgeDim, VertexDim
from icon4py.grid.horizontal import HorizontalMarkerIndex
from icon4py.testutils.fixtures import data_provider, setup_icon_data  # noqa


@pytest.mark.datatest
def test_horizontal_grid_cell_indices(icon_grid):
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.local(CellDim) - 1,
        HorizontalMarkerIndex.local(CellDim) - 1,
    ) == (
        20896,
        20896,
    )  # halo + 1

    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.local(CellDim),
        HorizontalMarkerIndex.local(CellDim),
    ) == (
        -1,
        20896,
    )  # halo in icon is (1,20896)
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.interior(CellDim),
        HorizontalMarkerIndex.interior(CellDim),
    ) == (
        4104,
        20896,
    )  # interior
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.interior(CellDim) + 1,
        HorizontalMarkerIndex.interior(CellDim) + 1,
    ) == (
        0,
        850,
    )  # lb+1
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 1,
    ) == (850, 1688)
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
        HorizontalMarkerIndex.lateral_boundary(CellDim) + 2,
    ) == (
        1688,
        2511,
    )  # lb+2
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
        HorizontalMarkerIndex.nudging(CellDim) - 1,
    ) == (
        2511,
        3316,
    )  # lb+3
    assert icon_grid.get_indices_from_to(
        CellDim,
        HorizontalMarkerIndex.nudging(CellDim),
        HorizontalMarkerIndex.nudging(CellDim),
    ) == (
        3316,
        4104,
    )  # nudging


@pytest.mark.datatest
def test_horizontal_edge_indices(icon_grid):
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.interior(EdgeDim),
        HorizontalMarkerIndex.interior(EdgeDim),
    ) == (6176, 31558)
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local(EdgeDim) - 2,
        HorizontalMarkerIndex.local(EdgeDim) - 2,
    ) == (31558, 31558)
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local(EdgeDim) - 1,
        HorizontalMarkerIndex.local(EdgeDim) - 1,
    ) == (31558, 31558)
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.local(EdgeDim),
        HorizontalMarkerIndex.local(EdgeDim),
    ) == (
        -1,
        31558,
    )  # halo in icon is  (1, 31558)
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.nudging(EdgeDim) + 1,
        HorizontalMarkerIndex.nudging(EdgeDim) + 1,
    ) == (
        5387,
        6176,
    )  # nudging +1
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.nudging(EdgeDim),
        HorizontalMarkerIndex.nudging(EdgeDim),
    ) == (
        4989,
        5387,
    )  # nudging
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7,
    ) == (
        4184,
        4989,
    )  # lb +7
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6,
    ) == (
        3777,
        4184,
    )  # lb +6
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5,
    ) == (
        2954,
        3777,
    )  # lb +5
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4,
    ) == (
        2538,
        2954,
    )  # lb +4
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3,
    ) == (
        1700,
        2538,
    )  # lb +3
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2,
    ) == (
        1278,
        1700,
    )  # lb +2
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1,
    ) == (
        428,
        1278,
    )  # lb +1
    assert icon_grid.get_indices_from_to(
        EdgeDim,
        HorizontalMarkerIndex.lateral_boundary(EdgeDim),
        HorizontalMarkerIndex.lateral_boundary(EdgeDim),
    ) == (
        0,
        428,
    )  # lb +0


@pytest.mark.datatest
def test_horizontal_vertex_indices(icon_grid):
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.end(VertexDim),
        HorizontalMarkerIndex.end(VertexDim),
    ) == (10663, 10663)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local(VertexDim),
        HorizontalMarkerIndex.local(VertexDim),
    ) == (-1, 10663)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.local(VertexDim) - 1,
        HorizontalMarkerIndex.local(VertexDim) - 1,
    ) == (10663, 10663)

    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim),
        HorizontalMarkerIndex.lateral_boundary(VertexDim),
    ) == (0, 428)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1,
    ) == (428, 850)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2,
    ) == (850, 1266)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3,
    ) == (1266, 1673)
    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4,
        HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4,
    ) == (1673, 2071)

    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.interior(VertexDim),
        HorizontalMarkerIndex.interior(VertexDim),
    ) == (2071, 10663)

    assert icon_grid.get_indices_from_to(
        VertexDim,
        HorizontalMarkerIndex.nudging(VertexDim),
        HorizontalMarkerIndex.nudging(VertexDim),
    ) == (10663, 10663)
