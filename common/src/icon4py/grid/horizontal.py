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
from abc import ABC
from dataclasses import dataclass
from typing import Final

from gt4py.next.common import Dimension, Field

from icon4py.common import dimension
from icon4py.common.dimension import CellDim, ECVDim, EdgeDim


class HorizontalMarkerIndex:
    """
    Handles constants indexing into the start_index and end_index fields.

     ICON uses a double indexing scheme for field indices marking the start and end of special
     grid zone: The constants defined here (from mo_impl_constants.f90 and mo_impl_constants_grf.f90)
     are the indices that are used to index into the start_idx and end_idx arrays
     provided by the grid file where for each dimension the start index of the horizontal
     "zones" are defined:
     f.ex. an inlined access of the field F: Field[[CellDim], double] at the starting point of the lateral boundary zone would be

     F[start_idx_c[_LATERAL_BOUNDARY_CELLS]


     ICON uses a custom index range from [ICON_INDEX_OFFSET... ] such that the index 0 marks the
     internal entities for _all_ dimensions (Cell, Edge, Vertex) that is why we define these
     additional INDEX_OFFSETs here in order to swap back to a 0 base python array.

    """

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
    _HALO_EDGES: Final[int] = _MIN_RL_EDGE_INT - 1 + _ICON_INDEX_OFFSET_EDGES
    _LOCAL_EDGES: Final[int] = _MIN_RL_EDGE_INT + _ICON_INDEX_OFFSET_EDGES
    _END_EDGES: Final[int] = 0

    _LATERAL_BOUNDARY_CELLS: Final[int] = 1 + _ICON_INDEX_OFFSET_CELLS
    _INTERIOR_CELLS: Final[int] = _ICON_INDEX_OFFSET_CELLS
    _NUDGING_CELLS: Final[int] = _GRF_BOUNDARY_WIDTH_CELL + 1 + _ICON_INDEX_OFFSET_CELLS
    _HALO_CELLS: Final[int] = _MIN_RL_CELL_INT - 1 + _ICON_INDEX_OFFSET_CELLS
    _LOCAL_CELLS: Final[int] = _MIN_RL_CELL_INT + _ICON_INDEX_OFFSET_CELLS
    _END_CELLS: Final[int] = 0

    _LATERAL_BOUNDARY_VERTICES = 1 + _ICON_INDEX_OFFSET_VERTEX
    _INTERIOR_VERTICES: Final[int] = _ICON_INDEX_OFFSET_VERTEX
    _NUDGING_VERTICES: Final[int] = 0
    _HALO_VERTICES: Final[int] = _MIN_RL_VERTEX_INT - 1 + _ICON_INDEX_OFFSET_VERTEX
    _LOCAL_VERTICES: Final[int] = _MIN_RL_VERTEX_INT + _ICON_INDEX_OFFSET_VERTEX
    _END_VERTICES: Final[int] = 0

    _lateral_boundary = {
        dimension.CellDim: _LATERAL_BOUNDARY_CELLS,
        dimension.EdgeDim: _LATERAL_BOUNDARY_EDGES,
        dimension.VertexDim: _LATERAL_BOUNDARY_VERTICES,
    }
    _local = {
        dimension.CellDim: _LOCAL_CELLS,
        dimension.EdgeDim: _LOCAL_EDGES,
        dimension.VertexDim: _LOCAL_VERTICES,
    }
    _halo = {
        dimension.CellDim: _HALO_CELLS,
        dimension.EdgeDim: _HALO_EDGES,
        dimension.VertexDim: _HALO_VERTICES,
    }
    _interior = {
        dimension.CellDim: _INTERIOR_CELLS,
        dimension.EdgeDim: _INTERIOR_EDGES,
        dimension.VertexDim: _INTERIOR_VERTICES,
    }
    _nudging = {
        dimension.CellDim: _NUDGING_CELLS,
        dimension.EdgeDim: _NUDGING_EDGES,
        dimension.VertexDim: _NUDGING_VERTICES,
    }
    _end = {
        dimension.CellDim: _END_CELLS,
        dimension.EdgeDim: _END_EDGES,
        dimension.VertexDim: _END_VERTICES,
    }

    @classmethod
    def lateral_boundary(cls, dim: Dimension) -> int:
        """Indicate lateral boundary.

        These points correspond to the sorted points in ICON, the marker can be incremented in order
        to access higher order boundary lines
        """
        return cls._lateral_boundary[dim]

    @classmethod
    def local(cls, dim: Dimension) -> int:
        """Indicate points that are owned by the processing unit, i.e. no halo points."""
        return cls._local[dim]

    @classmethod
    def halo(cls, dim: Dimension) -> int:
        return cls._halo[dim]

    @classmethod
    def nudging(cls, dim: Dimension) -> int:
        """Indicate the nudging zone."""
        return cls._nudging[dim]

    @classmethod
    def interior(cls, dim: Dimension) -> int:
        """Indicate interior i.e. unordered prognostic cells in ICON."""
        return cls._interior[dim]

    @classmethod
    def end(cls, dim: Dimension) -> int:
        return cls._end[dim]


class IconHorizontalDomainZone(ABC):
    def __init__(self, dim: Dimension):
        self._marker = HorizontalMarkerIndex.nudging(dim)

    def __call__(self, *args, **kwargs):
        return self._marker

    def __add__(self, other: int):
        return self._marker + other


class Nudging(IconHorizontalDomainZone):
    def __init__(self, dim: Dimension):
        super().__init__(dim)


class LateralBoundary(IconHorizontalDomainZone):
    def __init__(self, dim: Dimension):
        super().__init__(dim)


class Interior(IconHorizontalDomainZone):
    def __init__(self, dim: Dimension):
        super().__init__(dim)


def nudging(dim: CellDim, offset=0):
    return HorizontalMarkerIndex.nudging(dim) + offset


@dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


# TODO(Magdalena): allow initialization with only partial values
#  (becomes tedious for testing otherwise): hence this should
#  that should not be a data class
class EdgeParams:
    def __init__(
        self,
        tangent_orientation=None,
        primal_edge_lengths=None,
        inverse_primal_edge_lengths=None,
        dual_edge_lengths=None,
        inverse_dual_edge_lengths=None,
        inverse_vertex_vertex_lengths=None,
        primal_normal_vert_x=None,
        primal_normal_vert_y=None,
        dual_normal_vert_x=None,
        dual_normal_vert_y=None,
        edge_areas=None,
    ):

        self.tangent_orientation: Field[[EdgeDim], float] = tangent_orientation
        r"""
        Orientation of vector product of the edge and the adjacent cell centers
             v3
            /  \
           /    \
          /  c1  \
         /    |   \
         v1---|--->v2
         \    |   /
          \   v  /
           \ c2 /
            \  /
            v4
        +1 or -1 depending on whether the vector product of
        (v2-v1) x (c2-c1) points outside (+) or inside (-) the sphere

        defined in ICON in mo_model_domain.f90:t_grid_edges%tangent_orientation
        """

        self.primal_edge_lengths: Field[[EdgeDim], float] = primal_edge_lengths
        """
        Length of the triangle edge.

        defined int ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
        """

        self.inverse_primal_edge_lengths: Field[
            [EdgeDim], float
        ] = inverse_primal_edge_lengths
        """
        Inverse of the triangle edge length: 1.0/primal_edge_length.

        defined int ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
        """

        self.dual_edge_lengths: Field[[EdgeDim], float] = dual_edge_lengths
        """
        Length of the hexagon/pentagon edge.

        defined int ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
        """

        self.inverse_dual_edge_lengths: Field[
            [EdgeDim], float
        ] = inverse_dual_edge_lengths
        """
        Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

        defined int ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
        """

        self.inverse_vertex_vertex_lengths: Field[
            [EdgeDim], float
        ] = inverse_vertex_vertex_lengths
        r"""
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

        self.primal_normal_vert: tuple[
            Field[[ECVDim], float], Field[[ECVDim], float]
        ] = (primal_normal_vert_x, primal_normal_vert_y)
        """
        Normal of the triangle edge, projected onto the location of the vertices

        defined int ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
        """

        self.dual_normal_vert: tuple[Field[[ECVDim], float], Field[[ECVDim], float]] = (
            dual_normal_vert_x,
            dual_normal_vert_y,
        )
        """
        Tangent to the triangle edge, projected onto the location of vertices.

         defined int ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
        """

        self.edge_areas: Field[[EdgeDim], float] = edge_areas
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
