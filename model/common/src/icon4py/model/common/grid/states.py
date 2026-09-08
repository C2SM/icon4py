# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa


@dataclasses.dataclass(frozen=True, kw_only=True)
class EdgeParams:
    """Edge geometry of the grid, from ICON's ``t_grid_edges``.

    A member annotated ``| None`` is one that not every construction path can
    supply; everything else must be provided by all of them.
    """

    tangent_orientation: fa.EdgeField[float]
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

    inverse_primal_edge_lengths: fa.EdgeField[float]
    """
    Inverse of the triangle edge length: 1.0/primal_edge_length.

    defined in ICON in mo_model_domain.f90:t_grid_edges%inv_primal_edge_length
    """

    inverse_dual_edge_lengths: fa.EdgeField[float]
    """
    Inverse of hexagon/pentagon edge length: 1.0/dual_edge_length.

    defined in ICON in mo_model_domain.f90:t_grid_edges%inv_dual_edge_length
    """

    inverse_vertex_vertex_lengths: fa.EdgeField[float]
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

    primal_normal_vert: tuple[
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
    ]
    """
    Normal of the triangle edge, projected onto the location of the
    four vertices of neighboring cells.

    defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_vert
    and computed in ICON in mo_intp_coeffs.f90
    """

    dual_normal_vert: tuple[
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
    ]
    """
    zonal (x) and meridional (y) components of vector tangent to the triangle edge,
    projected onto the location of the four vertices of neighboring cells.

    defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_vert
    and computed in ICON in mo_intp_coeffs.f90
    """

    primal_normal_cell: tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]
    """
    zonal (x) and meridional (y) components of vector normal to the cell edge
    projected onto the location of neighboring cell centers.

    defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal_cell
    and computed in ICON in mo_intp_coeffs.f90
    """

    dual_normal_cell: tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]
    """
    zonal (x) and meridional (y) components of vector normal to the dual edge
    projected onto the location of neighboring cell centers.

    defined in ICON in mo_model_domain.f90:t_grid_edges%dual_normal_cell
    and computed in ICON in mo_intp_coeffs.f90
    """

    edge_areas: fa.EdgeField[float]
    """
    Area of the quadrilateral whose edges are the primal edge and
    the associated dual edge.

    defined in ICON in mo_model_domain.f90:t_grid_edges%area_edge
    and computed in ICON in mo_intp_coeffs.f90
    """

    coriolis_frequency: fa.EdgeField[float]
    """
    Declared as f_e in ICON. Coriolis parameter at cell edges.
    """

    edge_center: tuple[fa.EdgeField[float], fa.EdgeField[float]]
    """
    Latitude and longitude at the edge center

    defined in ICON in mo_model_domain.f90:t_grid_edges%center
    """

    primal_normal: tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]
    """
    zonal (x) and meridional (y) components of vector normal to the cell edge

    defined in ICON in mo_model_domain.f90:t_grid_edges%primal_normal
    """

    primal_edge_lengths: fa.EdgeField[float] | None = None
    """
    Length of the triangle edge.

    defined in ICON in mo_model_domain.f90:t_grid_edges%primal_edge_length
    """

    dual_edge_lengths: fa.EdgeField[float] | None = None
    """
    Length of the hexagon/pentagon edge.
    vertices of the hexagon/pentagon are cell centers and its center
    is located at the common vertex.
    the dual edge bisects the primal edge othorgonally.

    defined in ICON in mo_model_domain.f90:t_grid_edges%dual_edge_length
    """

    edge_cell_distances: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] | None = None
    """
    Distance between the edge midpoint and the circumcenters of the two adjacent cells.

    ICON's ``grid_init`` does not pass this field, so the Fortran bindings path
    leaves it unset.

    defined in ICON in mo_model_domain.f90:t_grid_edges%edge_cell_length
    """


@dataclasses.dataclass(frozen=True)
class CellParams:
    #: Latitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lat: fa.CellField[float]
    #: Longitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lon: fa.CellField[float]
    #: Area of a cell, defined in ICON in mo_model_domain.f90:t_grid_cells%area
    area: fa.CellField[float]
    #: Not supplied by every construction path.
    mean_cell_area: float | None = None
