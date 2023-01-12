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
from typing import Final

from functional.common import Dimension

from icon4py.common import dimension


class HorizontalMarkerIndex:
    NUM_GHOST_ROWS: Final[int] = 2
    # values from mo_impl_constants.f90
    _ICON_INDEX_OFFSET_CELLS: Final[int] = 8
    _GRF_BOUNDARY_WIDTH_CELL: Final[int] = 4
    _MIN_RL_CELL_INT: Final[int] = -4
    _MIN_RL_CELL: Final[int] = _MIN_RL_CELL_INT - 2 * NUM_GHOST_ROWS
    _MAX_RL_CELL: Final[int] = 5

    _ICON_INDEX_OFFSET_VERTEX: Final[int] = 7
    _MIN_RL_VERTEX_INT: Final[int] = _MIN_RL_CELL_INT
    _MIN_RL_VERTEX: Final[int] = _MIN_RL_VERTEX_INT - (NUM_GHOST_ROWS + 1)
    _MAX_RL_VERTEX: Final[int] = _MAX_RL_CELL

    _ICON_INDEX_OFFSET_EDGES: Final[int] = 13
    _GRF_BOUNDARY_WIDTH_EDGES: Final[int] = 9
    _MIN_RL_EDGE_INT: Final[int] = 2 * _MIN_RL_CELL_INT
    _MIN_RL_EDGE: Final[int] = _MIN_RL_EDGE_INT - (2 * NUM_GHOST_ROWS + 1)
    _MAX_RL_EDGE: Final[int] = 2 * _MAX_RL_CELL

    _LOCAL_BOUNDARY_EDGES: Final[int] = 1 + _ICON_INDEX_OFFSET_EDGES
    _INTERIOR_EDGES: Final[int] = _ICON_INDEX_OFFSET_EDGES
    _NUDGING_EDGES: Final[int] = _GRF_BOUNDARY_WIDTH_EDGES + _ICON_INDEX_OFFSET_EDGES
    _HALO_EDGES: Final[int] = _MIN_RL_EDGE_INT + _ICON_INDEX_OFFSET_EDGES
    _END_EDGES: Final[int] = 0

    _LOCAL_BOUNDARY_CELLS: Final[int] = 1 + _ICON_INDEX_OFFSET_CELLS
    _INTERIOR_CELLS: Final[int] = _ICON_INDEX_OFFSET_CELLS
    _NUDGING_CELLS: Final[int] = _GRF_BOUNDARY_WIDTH_CELL + 1 + _ICON_INDEX_OFFSET_CELLS
    _HALO_CELLS: Final[int] = _MIN_RL_CELL_INT + _ICON_INDEX_OFFSET_CELLS
    _END_CELLS: Final[int] = 0

    _LOCAL_BOUNDARY_VERTICES = 1 + _ICON_INDEX_OFFSET_VERTEX
    _INTERIOR_VERTICES: Final[int] = _ICON_INDEX_OFFSET_VERTEX
    _NUDGING_VERTICES: Final[int] = 0
    _HALO_VERTICES: Final[int] = _MIN_RL_VERTEX_INT + _ICON_INDEX_OFFSET_VERTEX
    _END_VERTICES: Final[int] = 0

    @classmethod
    def local_boundary(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._LOCAL_BOUNDARY_CELLS
            case (dimension.EdgeDim):
                return cls._LOCAL_BOUNDARY_EDGES
            case (dimension.VertexDim):
                return cls._LOCAL_BOUNDARY_VERTICES

    @classmethod
    def local(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._HALO_CELLS
            case (dimension.EdgeDim):
                return cls._HALO_EDGES
            case (dimension.VertexDim):
                return cls._HALO_VERTICES

    @classmethod
    def nudging(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._NUDGING_CELLS
            case (dimension.EdgeDim):
                return cls._NUDGING_EDGES
            case (dimension.VertexDim):
                return cls._NUDGING_VERTICES

    @classmethod
    def interior(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._INTERIOR_CELLS
            case (dimension.EdgeDim):
                return cls._INTERIOR_EDGES
            case (dimension.VertexDim):
                return cls._INTERIOR_VERTICES

    @classmethod
    def end(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._END_CELLS
            case (dimension.EdgeDim):
                return cls._END_EDGES
            case (dimension.VertexDim):
                return cls._END_VERTICES


class HorizontalMeshConfig:
    def __init__(self, num_vertices: int, num_edges: int, num_cells: int):
        self._num_vertices = num_vertices
        self._num_edges = num_edges
        self._num_cells = num_cells

    @property
    def num_vertices(self):
        return self._num_vertices

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def num_cells(self):
        return self._num_cells
