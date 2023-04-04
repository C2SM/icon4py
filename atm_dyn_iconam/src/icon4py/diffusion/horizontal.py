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
from dataclasses import dataclass
from typing import Final

from gt4py.next.common import Dimension, Field

from icon4py.common import dimension
from icon4py.common.dimension import CellDim, ECVDim, EdgeDim


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

    _LATERAL_BOUNDARY_EDGES: Final[int] = 1 + _ICON_INDEX_OFFSET_EDGES
    _INTERIOR_EDGES: Final[int] = _ICON_INDEX_OFFSET_EDGES
    _NUDGING_EDGES: Final[int] = _GRF_BOUNDARY_WIDTH_EDGES + _ICON_INDEX_OFFSET_EDGES
    _HALO_EDGES: Final[int] = _MIN_RL_EDGE_INT + _ICON_INDEX_OFFSET_EDGES
    _END_EDGES: Final[int] = 0

    _LATERAL_BOUNDARY_CELLS: Final[int] = 1 + _ICON_INDEX_OFFSET_CELLS
    _INTERIOR_CELLS: Final[int] = _ICON_INDEX_OFFSET_CELLS
    _NUDGING_CELLS: Final[int] = _GRF_BOUNDARY_WIDTH_CELL + 1 + _ICON_INDEX_OFFSET_CELLS
    _HALO_CELLS: Final[int] = _MIN_RL_CELL_INT + _ICON_INDEX_OFFSET_CELLS
    _END_CELLS: Final[int] = 0

    _LATERAL_BOUNDARY_VERTICES = 1 + _ICON_INDEX_OFFSET_VERTEX
    _INTERIOR_VERTICES: Final[int] = _ICON_INDEX_OFFSET_VERTEX
    _NUDGING_VERTICES: Final[int] = 0
    _HALO_VERTICES: Final[int] = _MIN_RL_VERTEX_INT + _ICON_INDEX_OFFSET_VERTEX
    _END_VERTICES: Final[int] = 0

    @classmethod
    def lateral_boundary(cls, dim: Dimension) -> int:
        match (dim):
            case (dimension.CellDim):
                return cls._LATERAL_BOUNDARY_CELLS
            case (dimension.EdgeDim):
                return cls._LATERAL_BOUNDARY_EDGES
            case (dimension.VertexDim):
                return cls._LATERAL_BOUNDARY_VERTICES

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


@dataclass(frozen=True)
class HorizontalMeshSize:
    num_vertices: int
    num_edges: int
    num_cells: int


@dataclass(frozen=True)
class EdgeParams:
    tangent_orientation: Field[[EdgeDim], float]
    """
    Orientation of vector product of the edge and the adjacent cell centers
         v3
        /  \
       /    \
      /  c1  \
     /    |   \
     v1----|--->v2
     \    |   /
      \   v  /
       \ c2 /
        \  /
        v4
    +1 or -1 depending on whether the vector product of
    (v2-v1) x (c2-c1) points outside (+) or inside (-) the sphere

    defined in ICON in mo_model_domain.f90:t_grid_edges%tangent_orientation
    """

    primal_edge_lengths: Field[[EdgeDim], float]
    """
    Length of the triangle edge.

    defined int ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
    """

    inverse_primal_edge_lengths: Field[[EdgeDim], float]
    """
    Inverse of the triangle edge length: 1.0/primal_edge_length.

    defined int ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
    """

    dual_edge_lengths: Field[[EdgeDim], float]
    """
    Length of the hexagon/pentagon edge.

    defined int ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
    """

    inverse_dual_edge_lengths: Field[[EdgeDim], float]
    """
    Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

    defined int ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
    """

    inverse_vertex_vertex_lengths: Field[[EdgeDim], float]
    """
    Inverse distance between outer vertices of adjacent cells.

    v1--------
    |       /|
    |      / |
    |    e   |
    |  /     |
    |/       |
    --------v2

    inverse_vertex_vertex_length(e) = 1.0/|v2-v1|

    defined int ICON in mo_model_domain.f90:t_grid_edges%inv_vert_vert_length
    """

    primal_normal_vert: tuple[Field[[ECVDim], float], Field[[ECVDim], float]]
    """
    Normal of the triangle edge, projected onto the location of the vertices

    defined int ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
    """

    dual_normal_vert: tuple[Field[[ECVDim], float], Field[[ECVDim], float]]
    """
    Tangent to the triangle edge, projected onto the location of vertices.

     defined int ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
    """

    edge_areas: Field[[EdgeDim], float]
    """
    Area of the quadrilateral (two triangles) adjacent to the edge.

    defined int ICON in mo_model_domain.f90:t_grid_edges%area_edge
    """


@dataclass(frozen=True)
class CellParams:
    area: Field[[CellDim], float]
    """
    Area of a cell.

    defined int ICON in mo_model_domain.f90:t_grid_cells%area
    """
