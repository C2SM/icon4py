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
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(CellDim), 4104),
    (HorizontalMarkerIndex.interior(CellDim) + 1, 0),
    (HorizontalMarkerIndex.local(CellDim) - 1, 20896),
    (HorizontalMarkerIndex.local(CellDim), -1), # halo in icon is (1,20896)
    (HorizontalMarkerIndex.nudging(CellDim), 3316),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 2511),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 1688),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 850),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 0)
                                           ])
def test_horizontal_cell_start_indices(icon_grid, marker, index):
    assert index == icon_grid.get_start_index(CellDim, marker)

@pytest.mark.datatest
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(CellDim), 20896),
    (HorizontalMarkerIndex.interior(CellDim) + 1, 850),
    ( HorizontalMarkerIndex.local(CellDim) - 2,20896 ),
    (HorizontalMarkerIndex.local(CellDim) - 1, 20896),
    (HorizontalMarkerIndex.local(CellDim), 31558),
    (HorizontalMarkerIndex.nudging(CellDim), 4104),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 3, 3316),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 2, 2511),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 1, 1688),
    (HorizontalMarkerIndex.lateral_boundary(CellDim) + 0, 428),])
def test_horizontal_edge_end_indices(icon_grid, marker, index):
    assert index == icon_grid.get_end_index(CellDim, marker)

@pytest.mark.datatest
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(EdgeDim), 6176),
    ( HorizontalMarkerIndex.local(EdgeDim) - 2,31558 ),
    (HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
    (HorizontalMarkerIndex.local(EdgeDim), -1),
    (HorizontalMarkerIndex.nudging(EdgeDim) + 1,5387),
    (HorizontalMarkerIndex.nudging(EdgeDim), 4989),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7,4184 ),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 3777),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 2954),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2538),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 1700),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1278),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 428),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 0)
                                           ])
def test_horizontal_edge_start_indices(icon_grid, marker, index):
    assert index == icon_grid.get_start_index(EdgeDim, marker)

@pytest.mark.datatest
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(EdgeDim), 31558),
    ( HorizontalMarkerIndex.local(EdgeDim) - 2,31558 ),
    (HorizontalMarkerIndex.local(EdgeDim) - 1, 31558),
    (HorizontalMarkerIndex.local(EdgeDim), 31558),
    (HorizontalMarkerIndex.nudging(EdgeDim) + 1,6176),
    (HorizontalMarkerIndex.nudging(EdgeDim), 5387),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 7, 4989),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 6, 4184),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 5, 3777),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 4, 2954),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 3, 2538),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 2, 1700),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1, 1278),
    (HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 0, 428),])
def test_horizontal_edge_end_indices(icon_grid, marker, index):
    assert index == icon_grid.get_end_index(EdgeDim, marker)


@pytest.mark.datatest
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(VertexDim), 2071),
    (HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
    (HorizontalMarkerIndex.local(VertexDim), -1),
    (HorizontalMarkerIndex.nudging(VertexDim) + 1, 10663),
    (HorizontalMarkerIndex.nudging(VertexDim), 10663),
    (HorizontalMarkerIndex.end(VertexDim), 10663),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 1673),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1266),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 850),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 428),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 0)
                                           ])
def test_horizontal_vertex_start_indices(icon_grid, marker, index):
    assert index == icon_grid.get_start_index(VertexDim, marker)

@pytest.mark.datatest
@pytest.mark.parametrize( "marker, index",[
    ( HorizontalMarkerIndex.interior(VertexDim), 10663),
    ( HorizontalMarkerIndex.local(VertexDim) - 2,10663 ),
    (HorizontalMarkerIndex.local(VertexDim) - 1, 10663),
    (HorizontalMarkerIndex.local(VertexDim), 10663),
    (HorizontalMarkerIndex.nudging(VertexDim) + 1,10663),
    (HorizontalMarkerIndex.nudging(VertexDim), 10663),
    (HorizontalMarkerIndex.end(VertexDim), 10663),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 4, 2071),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 3, 1673),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 2, 1266),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 1, 850),
    (HorizontalMarkerIndex.lateral_boundary(VertexDim) + 0, 428),])
def test_horizontal_vertex_end_indices(icon_grid, marker, index):
    assert index == icon_grid.get_end_index(VertexDim, marker)


