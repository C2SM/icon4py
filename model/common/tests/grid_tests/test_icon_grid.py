# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex


@pytest.mark.datatest
# TODO(Magdalena) HorizontalMarkerIndex.local(dim) does not yield equivalent results form grid file
#  and serialized data, why?. Serialized data has those strange -1 values
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (dims.CellDim, HorizontalMarkerIndex.interior(dims.CellDim), 20896),
        (dims.CellDim, HorizontalMarkerIndex.interior(dims.CellDim) + 1, 850),
        (dims.CellDim, HorizontalMarkerIndex.local(dims.CellDim) - 2, 20896),
        (dims.CellDim, HorizontalMarkerIndex.local(dims.CellDim) - 1, 20896),
        (dims.CellDim, HorizontalMarkerIndex.local(dims.CellDim), 20896),
        (dims.CellDim, HorizontalMarkerIndex.nudging(dims.CellDim), 4104),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 3, 3316),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 2, 2511),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1, 1688),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 0, 850),
        (dims.EdgeDim, HorizontalMarkerIndex.interior(dims.EdgeDim), 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim) - 2, 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim) - 1, 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim), 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.nudging(dims.EdgeDim) + 1, 6176),
        (dims.EdgeDim, HorizontalMarkerIndex.nudging(dims.EdgeDim), 5387),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 7, 4989),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 6, 4184),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 5, 3777),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 4, 2954),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 3, 2538),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 2, 1700),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 1, 1278),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 0, 428),
        (dims.VertexDim, HorizontalMarkerIndex.interior(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.local(dims.VertexDim) - 2, 10663),
        (dims.VertexDim, HorizontalMarkerIndex.local(dims.VertexDim) - 1, 10663),
        (dims.VertexDim, HorizontalMarkerIndex.local(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.nudging(dims.VertexDim) + 1, 10663),
        (dims.VertexDim, HorizontalMarkerIndex.nudging(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.end(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 4, 2071),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 3, 1673),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 2, 1266),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 1, 850),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 0, 428),
    ],
)
def test_horizontal_end_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_end_index(dim, marker)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (dims.CellDim, HorizontalMarkerIndex.interior(dims.CellDim), 4104),
        (dims.CellDim, HorizontalMarkerIndex.interior(dims.CellDim) + 1, 0),
        (dims.CellDim, HorizontalMarkerIndex.local(dims.CellDim) - 1, 20896),
        (dims.CellDim, HorizontalMarkerIndex.local(dims.CellDim), -1),
        (dims.CellDim, HorizontalMarkerIndex.halo(dims.CellDim), 20896),
        (dims.CellDim, HorizontalMarkerIndex.nudging(dims.CellDim), 3316),
        (dims.CellDim, HorizontalMarkerIndex.nudging(dims.CellDim) - 1, 2511),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 3, 2511),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 2, 1688),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1, 850),
        (dims.CellDim, HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 0, 0),
        (dims.EdgeDim, HorizontalMarkerIndex.interior(dims.EdgeDim), 6176),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim) - 2, 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim) - 1, 31558),
        (dims.EdgeDim, HorizontalMarkerIndex.local(dims.EdgeDim), -1),  # ????
        (dims.EdgeDim, HorizontalMarkerIndex.nudging(dims.EdgeDim) + 1, 5387),
        (dims.EdgeDim, HorizontalMarkerIndex.nudging(dims.EdgeDim), 4989),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 7, 4184),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 6, 3777),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 5, 2954),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 4, 2538),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 3, 1700),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 2, 1278),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 1, 428),
        (dims.EdgeDim, HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 0, 0),
        (dims.VertexDim, HorizontalMarkerIndex.interior(dims.VertexDim), 2071),
        (dims.VertexDim, HorizontalMarkerIndex.local(dims.VertexDim) - 1, 10663),
        (dims.VertexDim, HorizontalMarkerIndex.local(dims.VertexDim), -1),  # ???
        (dims.VertexDim, HorizontalMarkerIndex.nudging(dims.VertexDim) + 1, 10663),
        (dims.VertexDim, HorizontalMarkerIndex.nudging(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.end(dims.VertexDim), 10663),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 4, 1673),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 3, 1266),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 2, 850),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 1, 428),
        (dims.VertexDim, HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 0, 0),
    ],
)
def test_horizontal_start_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_start_index(dim, marker)


@pytest.mark.datatest
def test_grid_size(grid_savepoint):
    assert 10663 == grid_savepoint.num(dims.VertexDim)
    assert 20896 == grid_savepoint.num(dims.CellDim)
    assert 31558 == grid_savepoint.num(dims.EdgeDim)
