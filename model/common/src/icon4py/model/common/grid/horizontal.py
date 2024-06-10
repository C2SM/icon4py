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
import math
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Final

from gt4py.next import Dimension, Field

from icon4py.model.common import constants, dimension
from icon4py.model.common.dimension import (
    CellDim,
    ECDim,
    ECVDim,
    EdgeDim,
)


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
_GRF_NUDGEZONE_START_EDGES: Final[int] = _GRF_BOUNDARY_WIDTH_EDGES + 1
_GRF_NUDGEZONE_WIDTH: Final[int] = 8
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

    _lateral_boundary: ClassVar = {
        dimension.CellDim: _LATERAL_BOUNDARY_CELLS,
        dimension.EdgeDim: _LATERAL_BOUNDARY_EDGES,
        dimension.VertexDim: _LATERAL_BOUNDARY_VERTICES,
    }
    _local: ClassVar = {
        dimension.CellDim: _LOCAL_CELLS,
        dimension.EdgeDim: _LOCAL_EDGES,
        dimension.VertexDim: _LOCAL_VERTICES,
    }
    _halo: ClassVar = {
        dimension.CellDim: _HALO_CELLS,
        dimension.EdgeDim: _HALO_EDGES,
        dimension.VertexDim: _HALO_VERTICES,
    }
    _interior: ClassVar = {
        dimension.CellDim: _INTERIOR_CELLS,
        dimension.EdgeDim: _INTERIOR_EDGES,
        dimension.VertexDim: _INTERIOR_VERTICES,
    }
    _nudging: ClassVar = {
        dimension.CellDim: _NUDGING_CELLS,
        dimension.EdgeDim: _NUDGING_EDGES,
        # TODO [magdalena] there is no nudging for vertices?
        dimension.VertexDim: _NUDGING_VERTICES,
    }
    _end: ClassVar = {
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
    def nudging_2nd_level(cls, dim: Dimension) -> int:
        """Indicate the nudging zone for 2nd level."""
        return cls.nudging(dim) + 1

    @classmethod
    def interior(cls, dim: Dimension) -> int:
        """Indicate interior i.e. unordered prognostic cells in ICON."""
        return cls._interior[dim]

    @classmethod
    def end(cls, dim: Dimension) -> int:
        return cls._end[dim]


@dataclass(frozen=True)
class HorizontalGridSize:
    num_vertices: int
    num_edges: int
    num_cells: int


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
        primal_normal_cell_x=None,
        dual_normal_cell_x=None,
        primal_normal_cell_y=None,
        dual_normal_cell_y=None,
        edge_areas=None,
        f_e=None,
        edge_center_lat=None,
        edge_center_lon=None,
        primal_normal_x=None,
        primal_normal_y=None,
    ):
        self.tangent_orientation: Field[[EdgeDim], float] = tangent_orientation
        """
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

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
        """

        self.inverse_primal_edge_lengths: Field[[EdgeDim], float] = inverse_primal_edge_lengths
        """
        Inverse of the triangle edge length: 1.0/primal_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
        """

        self.dual_edge_lengths: Field[[EdgeDim], float] = dual_edge_lengths
        """
        Length of the hexagon/pentagon edge.
        vertices of the hexagon/pentagon are cell centers and its center
        is located at the common vertex.
        the dual edge bisects the primal edge othorgonally.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
        """

        self.inverse_dual_edge_lengths: Field[[EdgeDim], float] = inverse_dual_edge_lengths
        """
        Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
        """

        self.inverse_vertex_vertex_lengths: Field[[EdgeDim], float] = inverse_vertex_vertex_lengths
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

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_vert_vert_length
        """

        self.primal_normal_vert: tuple[Field[[ECVDim], float], Field[[ECVDim], float]] = (
            primal_normal_vert_x,
            primal_normal_vert_y,
        )
        """
        Normal of the triangle edge, projected onto the location of the
        four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_vert: tuple[Field[[ECVDim], float], Field[[ECVDim], float]] = (
            dual_normal_vert_x,
            dual_normal_vert_y,
        )
        """
        zonal (x) and meridional (y) components of vector tangent to the triangle edge,
        projected onto the location of the four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.primal_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]] = (
            primal_normal_cell_x,
            primal_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_cell: tuple[Field[[ECDim], float], Field[[ECDim], float]] = (
            dual_normal_cell_x,
            dual_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the dual edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.edge_areas: Field[[EdgeDim], float] = edge_areas
        """
        Area of the quadrilateral whose edges are the primal edge and
        the associated dual edge.

        defined in ICON in mo_model_domain.f90:t_grid_edges%area_edge
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.f_e: Field[[EdgeDim], float] = f_e
        """
        Coriolis parameter at cell edges
        """

        self.edge_center: tuple[Field[[EdgeDim], float], Field[[EdgeDim], float]] = (
            edge_center_lat,
            edge_center_lon,
        )
        """
        Latitude and longitude at the edge center

        defined in ICON in mo_model_domain.f90:t_grid_edges%center
        """

        self.primal_normal: tuple[Field[[ECDim], float], Field[[ECDim], float]] = (
            primal_normal_x,
            primal_normal_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal
        """


@dataclass(frozen=True)
class CellParams:
    #: Latitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lat: Field[[CellDim], float] = None
    #: Longitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lon: Field[[CellDim], float] = None
    #: Area of a cell, defined in ICON in mo_model_domain.f90:t_grid_cells%area
    area: Field[[CellDim], float] = None
    #: Mean area of a cell [m^2] = total surface area / numer of cells defined in ICON in in mo_model_domimp_patches.f90
    mean_cell_area: float = None
    length_rescale_factor: float = 1.0

    @classmethod
    def from_global_num_cells(
        cls,
        cell_center_lat: Field[[CellDim], float],
        cell_center_lon: Field[[CellDim], float],
        area: Field[[CellDim], float],
        global_num_cells: int,
        length_rescale_factor: float = 1.0,
    ):
        mean_cell_area = cls._compute_mean_cell_area(constants.EARTH_RADIUS, global_num_cells)
        return cls(
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            area=area,
            mean_cell_area=mean_cell_area,
            length_rescale_factor=length_rescale_factor,
        )

    @cached_property
    def characteristic_length(self):
        return math.sqrt(self.mean_cell_area)

    @cached_property
    def mean_cell_area(self):
        return self.mean_cell_area

    @staticmethod
    def _compute_mean_cell_area(radius, num_cells):
        """
        Compute the mean cell area.

        Computes the mean cell area by dividing the sphere by the number of cells in the
        global grid.

        Args:
            radius: average earth radius, might be rescaled by a scaling parameter
            num_cells: number of cells on the global grid
        Returns: mean area of one cell [m^2]
        """
        return 4.0 * math.pi * radius**2 / num_cells


class RefinCtrlLevel:
    _boundary_nudging_start: ClassVar = {
        EdgeDim: _GRF_BOUNDARY_WIDTH_EDGES + 1,
        CellDim: _GRF_BOUNDARY_WIDTH_CELL + 1,
    }

    @classmethod
    def boundary_nudging_start(cls, dim: Dimension) -> int:
        """Start refin_ctrl levels for boundary nudging (as seen from the child domain)."""
        try:
            return cls._boundary_nudging_start[dim]
        except KeyError as err:
            raise ValueError(
                f"nudging start level only exists for {CellDim} and {EdgeDim}"
            ) from err


