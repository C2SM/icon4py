# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import math

from gt4py import next as gtx

from icon4py.model.common import constants, dimension as dims, field_type_aliases as fa


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
        self.tangent_orientation: fa.EdgeField[float] = tangent_orientation
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

        self.primal_edge_lengths: fa.EdgeField[float] = primal_edge_lengths
        """
        Length of the triangle edge.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
        """

        self.inverse_primal_edge_lengths: fa.EdgeField[float] = inverse_primal_edge_lengths
        """
        Inverse of the triangle edge length: 1.0/primal_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
        """

        self.dual_edge_lengths: fa.EdgeField[float] = dual_edge_lengths
        """
        Length of the hexagon/pentagon edge.
        vertices of the hexagon/pentagon are cell centers and its center
        is located at the common vertex.
        the dual edge bisects the primal edge othorgonally.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
        """

        self.inverse_dual_edge_lengths: fa.EdgeField[float] = inverse_dual_edge_lengths
        """
        Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

        defined in ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
        """

        self.inverse_vertex_vertex_lengths: fa.EdgeField[float] = inverse_vertex_vertex_lengths
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

        self.primal_normal_vert: tuple[
            gtx.Field[[dims.ECVDim], float], gtx.Field[[dims.ECVDim], float]
        ] = (
            primal_normal_vert_x,
            primal_normal_vert_y,
        )
        """
        Normal of the triangle edge, projected onto the location of the
        four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_vert: tuple[
            gtx.Field[[dims.ECVDim], float], gtx.Field[[dims.ECVDim], float]
        ] = (
            dual_normal_vert_x,
            dual_normal_vert_y,
        )
        """
        zonal (x) and meridional (y) components of vector tangent to the triangle edge,
        projected onto the location of the four vertices of neighboring cells.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.primal_normal_cell: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            primal_normal_cell_x,
            primal_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.dual_normal_cell: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            dual_normal_cell_x,
            dual_normal_cell_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the dual edge
        projected onto the location of neighboring cell centers.

        defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_cell
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.edge_areas: fa.EdgeField[float] = edge_areas
        """
        Area of the quadrilateral whose edges are the primal edge and
        the associated dual edge.

        defined in ICON in mo_model_domain.f90:t_grid_edges%area_edge
        and computed in ICON in mo_intp_coeffs.f90
        """

        self.f_e: fa.EdgeField[float] = f_e
        """
        Coriolis parameter at cell edges
        """

        self.edge_center: tuple[fa.EdgeField[float], fa.EdgeField[float]] = (
            edge_center_lat,
            edge_center_lon,
        )
        """
        Latitude and longitude at the edge center

        defined in ICON in mo_model_domain.f90:t_grid_edges%center
        """

        self.primal_normal: tuple[
            gtx.Field[[dims.ECDim], float], gtx.Field[[dims.ECDim], float]
        ] = (
            primal_normal_x,
            primal_normal_y,
        )
        """
        zonal (x) and meridional (y) components of vector normal to the cell edge

        defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal
        """


@dataclasses.dataclass(frozen=True)
class CellParams:
    #: Latitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lat: fa.CellField[float] = None
    #: Longitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lon: fa.CellField[float] = None
    #: Area of a cell, defined in ICON in mo_model_domain.f90:t_grid_cells%area
    area: fa.CellField[float] = None
    #: Mean area of a cell [m^2] = total surface area / numer of cells defined in ICON in in mo_model_domimp_patches.f90
    mean_cell_area: float = None
    length_rescale_factor: float = 1.0

    @classmethod
    def from_global_num_cells(
        cls,
        cell_center_lat: fa.CellField[float],
        cell_center_lon: fa.CellField[float],
        area: fa.CellField[float],
        global_num_cells: int,
        length_rescale_factor: float = 1.0,
    ):
        if global_num_cells == 0:
            # Compute from the area array (should be a torus grid)
            # TODO (Magdalena) this would not work for a distributed setup (at
            # least not for a sphere) for the torus it would because cell area
            # is constant.
            mean_cell_area = area.asnumpy().mean()
        else:
            mean_cell_area = compute_mean_cell_area_for_sphere(constants.EARTH_RADIUS, global_num_cells)
        return cls(
            cell_center_lat=cell_center_lat,
            cell_center_lon=cell_center_lon,
            area=area,
            mean_cell_area=mean_cell_area,
            length_rescale_factor=length_rescale_factor,
        )

    @functools.cached_property
    def characteristic_length(self):
        return math.sqrt(self.mean_cell_area)

    @functools.cached_property
    def mean_cell_area(self):
        return self.mean_cell_area

def compute_mean_cell_area_for_sphere(radius, num_cells):
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


