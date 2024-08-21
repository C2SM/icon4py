# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid


@pytest.mark.datatest
# TODO(Magdalena) HorizontalMarkerIndex.local(dim) does not yield equivalent results form grid file
#  and serialized data, why?. Serialized data has those strange -1 values
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (dims.CellDim, h_grid.HorizontalMarkerIndex.interior(dims.CellDim), 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.interior(dims.CellDim) + 1, 850),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.local(dims.CellDim) - 2, 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.local(dims.CellDim) - 1, 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.local(dims.CellDim), 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.nudging(dims.CellDim), 4104),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 3, 3316),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 2, 2511),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1, 1688),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 0, 850),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.interior(dims.EdgeDim), 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim) - 2, 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim) - 1, 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim), 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.nudging(dims.EdgeDim) + 1, 6176),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.nudging(dims.EdgeDim), 5387),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 7, 4989),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 6, 4184),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 5, 3777),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 4, 2954),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 3, 2538),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 2, 1700),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 1, 1278),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 0, 428),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.interior(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.local(dims.VertexDim) - 2, 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.local(dims.VertexDim) - 1, 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.local(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.nudging(dims.VertexDim) + 1, 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.nudging(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.end(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 4, 2071),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 3, 1673),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 2, 1266),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 1, 850),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 0, 428),
    ],
)
def test_horizontal_end_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_end_index(dim, marker)
    
@pytest.mark.parametrize("dim, size", [(dims.CellDim, 20896), (dims.EdgeDim, 31558), (dims.VertexDim, 10663) ])   
def test_end(dim, size, icon_grid):
    assert icon_grid.end(dim) == size
       
@pytest.mark.parametrize("dim, size", [(dims.CellDim, 20896), (dims.EdgeDim, 31558), (dims.VertexDim, 10663) ])
@pytest.mark.parametrize("line", [h_grid.HaloLine.FIRST, h_grid.HaloLine.SECOND])
@pytest.mark.parametrize("index_type", [h_grid.IndexType.START, h_grid.IndexType.END])
def test_halo(icon_grid, dim, size, line, index_type ):
    # for single node this returns an empty region - start and end index are the same
    assert icon_grid.halo(dim, index_type, line) == size

def boundary_lines():
    for line in h_grid.BoundaryLine.__members__.values():
        yield line

@pytest.mark.parametrize("dim, size",
                         [(dims.CellDim, 20896), (dims.EdgeDim, 31558), (dims.VertexDim, 10663)])
def test_local(icon_grid, dim, size):  
    assert icon_grid.local(dim, h_grid.IndexType.START) == 0
    assert icon_grid.local(dim, h_grid.IndexType.END) == size

LATERAL_BOUNDARY_IDX ={
    dims.CellDim: [0, 850, 1688, 2511, 3316, 4104 ],
    dims.EdgeDim: [0, 428, 1278, 1700, 2538, 2954, 3777, 4184, 4989, 5387, 6176],
    dims.VertexDim:[0, 428, 850, 1266, 1673,2071]
}

@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
@pytest.mark.parametrize("line", [h_grid.BoundaryLine.FIRST, h_grid.BoundaryLine.SECOND, h_grid.BoundaryLine.THIRD, h_grid.BoundaryLine.FOURTH])
def test_lateral_boundary(icon_grid, dim, line):
    start_index  = icon_grid.lateral_boundary(dim, h_grid.IndexType.START, line)
    end_index = icon_grid.lateral_boundary(dim, h_grid.IndexType.END, line)
    assert start_index == LATERAL_BOUNDARY_IDX[dim][line.value]
    assert end_index == LATERAL_BOUNDARY_IDX[dim][line.value + 1]


@pytest.mark.parametrize("line", [h_grid.BoundaryLine.FIFTH, h_grid.BoundaryLine.SIXTH,
                                  h_grid.BoundaryLine.SEVENTH])
def test_lateral_boundary_higher_lines_for_edges(icon_grid, line):
    start_index = icon_grid.lateral_boundary(dims.EdgeDim, h_grid.IndexType.START, line)
    end_index = icon_grid.lateral_boundary(dims.EdgeDim, h_grid.IndexType.END, line)
    assert start_index == LATERAL_BOUNDARY_IDX[dims.EdgeDim][line.value]
    assert end_index == LATERAL_BOUNDARY_IDX[dims.EdgeDim][line.value + 1]

@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim])
@pytest.mark.parametrize("line", [h_grid.BoundaryLine.FIFTH, h_grid.BoundaryLine.SIXTH,
                                  h_grid.BoundaryLine.SEVENTH])
def test_lateral_boundary_higher_lines_cell_and_vertex(icon_grid, dim, line):
    with pytest.raises(ValueError) as e:
        icon_grid.lateral_boundary(dim, h_grid.IndexType.START, line)
        e.match(f"Invalid line number '{line}' and dimension '{dim}'")
        
        
NUDGING_IDX={
    dims.CellDim: [3316, 4104],
    dims.EdgeDim: [4989, 5387, 6176],
   
}
@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim])
def test_nudging(icon_grid, dim):
    start_index = icon_grid.nudging(dim, h_grid.IndexType.START)
    end_index = icon_grid.nudging(dim, h_grid.IndexType.END)
    assert start_index == NUDGING_IDX[dim][0]
    assert end_index == NUDGING_IDX[dim][1]
    
def test_nudging_second_line_for_edges(icon_grid):
    assert NUDGING_IDX[dims.EdgeDim][1] == icon_grid.nudging(dims.EdgeDim, h_grid.IndexType.START, line=h_grid.NudgingLine.SECOND)
    assert NUDGING_IDX[dims.EdgeDim][2] == icon_grid.nudging(dims.EdgeDim, h_grid.IndexType.END,
                                               line=h_grid.NudgingLine.SECOND)
def test_nudging_second_line_for_cells(icon_grid):
    with pytest.raises(ValueError) as e:
        icon_grid.nudging(dims.CellDim, h_grid.IndexType.START, line=h_grid.NudgingLine.SECOND)
        e.match("Invalid line number 'SECOND' and dimension 'CellDim")
    
def test_nudging_vertex(icon_grid):
    with pytest.raises(AssertionError) as e:
        icon_grid.nudging(dims.VertexDim, h_grid.IndexType.START)
        e.match("Invalid dimension for nudging 'VertexDim")

@pytest.mark.parametrize("dim", [dims.CellDim, dims.EdgeDim, dims.VertexDim])
def test_interior(icon_grid, dim):
    start_index = icon_grid.interior(dim, h_grid.IndexType.START)
    end_index = icon_grid.interior(dim, h_grid.IndexType.END)
    assert start_index == LATERAL_BOUNDARY_IDX[dim][-1]
    assert end_index == icon_grid.halo(dim, index_type=h_grid.IndexType.START, line_number=h_grid.HaloLine.FIRST)

@pytest.mark.datatest
@pytest.mark.parametrize(
    "dim, marker, index",
    [
        (dims.CellDim, h_grid.HorizontalMarkerIndex.interior(dims.CellDim), 4104),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.interior(dims.CellDim) + 1, 0),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.local(dims.CellDim) - 1, 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.local(dims.CellDim), -1),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.halo(dims.CellDim), 20896),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.nudging(dims.CellDim), 3316),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.nudging(dims.CellDim) - 1, 2511),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 3, 2511),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 2, 1688),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 1, 850),
        (dims.CellDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.CellDim) + 0, 0),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.interior(dims.EdgeDim), 6176),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim) - 2, 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim) - 1, 31558),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.local(dims.EdgeDim), -1),  # ????
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.nudging(dims.EdgeDim) + 1, 5387),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.nudging(dims.EdgeDim), 4989),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 7, 4184),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 6, 3777),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 5, 2954),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 4, 2538),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 3, 1700),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 2, 1278),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 1, 428),
        (dims.EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.EdgeDim) + 0, 0),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.interior(dims.VertexDim), 2071),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.local(dims.VertexDim) - 1, 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.local(dims.VertexDim), -1),  # ???
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.nudging(dims.VertexDim) + 1, 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.nudging(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.end(dims.VertexDim), 10663),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 4, 1673),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 3, 1266),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 2, 850),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 1, 428),
        (dims.VertexDim, h_grid.HorizontalMarkerIndex.lateral_boundary(dims.VertexDim) + 0, 0),
    ],
)
def test_horizontal_start_index(icon_grid, dim, marker, index):
    assert index == icon_grid.get_start_index(dim, marker)


@pytest.mark.datatest
def test_grid_size(grid_savepoint):
    assert 10663 == grid_savepoint.num(dims.VertexDim)
    assert 20896 == grid_savepoint.num(dims.CellDim)
    assert 31558 == grid_savepoint.num(dims.EdgeDim)
