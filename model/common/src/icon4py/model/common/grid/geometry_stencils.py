# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next import sin, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2V, E2V, EdgeDim
from icon4py.model.common.math.helpers import (
    arc_length,
    cross_product,
    geographical_to_cartesian_on_edges,
    geographical_to_cartesian_on_vertex,
    normalize_cartesian_vector,
    zonal_and_meridional_components_on_edges,
)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_of_edge_tangent(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Compute normalized cartesian vector tangential to an edge.

    That is: computes the distance between the two vertices adjacent to the edge:
    t = d(v1, v2)

    Args:
        vertex_lat: latitude of vertices
        vertex_lon: longitude of vertices
        edge_orientation: encoding of the edge orientation: (-1, +1) depending on whether the
            edge is directed from first to second neighbor of vice versa.
    Returns:
          x: x coordinate of normalized tangent vector
          y: y coordinate of normalized tangent vector
          z: z coordinate of normalized tangent vector
    """
    vertex_x, vertex_y, vertex_z = geographical_to_cartesian_on_vertex(vertex_lat, vertex_lon)

    x = edge_orientation * (vertex_x(E2V[1]) - vertex_x(E2V[0]))
    y = edge_orientation * (vertex_y(E2V[1]) - vertex_y(E2V[0]))
    z = edge_orientation * (vertex_z(E2V[1]) - vertex_z(E2V[0]))

    return normalize_cartesian_vector(x, y, z)


@gtx.field_operator
def cartesian_coordinates_of_edge_normal(
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_tangent_x: fa.EdgeField[ta.wpfloat],
    edge_tangent_y: fa.EdgeField[ta.wpfloat],
    edge_tangent_z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute the normal to the edge tangent vector.

    The normal is  the cross product of the edge center (cartesian vector) and the edge tangent (cartesian_vector): (edge_center) x (edge_tangent)

    Args:
        edge_lat: latitude of edge center
        edge_lon: longitude of edge center
        edge_tangent_x: x coordinate of edge tangent
        edge_tangent_y: y coordinate of edge tangent
        edge_tangent_z: z coordinate of edge tangent
    Returns:
        edge_normal_x: x coordinate of the normal
        edge_normal_y: y coordinate of the normal
        edge_normal_z: z coordinate of the normal
    """
    edge_center_x, edge_center_y, edge_center_z = geographical_to_cartesian_on_edges(
        edge_lat, edge_lon
    )
    x, y, z = cross_product(
        edge_center_x, edge_tangent_x, edge_center_y, edge_tangent_y, edge_center_z, edge_tangent_z
    )
    return normalize_cartesian_vector(x, y, z)


@gtx.field_operator
def cartesian_coordinates_edge_tangent_and_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute normalized cartesian vectors of edge tangent and edge normal."""
    tangent_x, tangent_y, tangent_z = cartesian_coordinates_of_edge_tangent(
        vertex_lat, vertex_lon, edge_orientation
    )
    normal_x, normal_y, normal_z = cartesian_coordinates_of_edge_normal(
        edge_lat,
        edge_lon,
        tangent_x,
        tangent_y,
        tangent_z,
    )

    return tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_of_edge_tangent_and_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
    tangent_x: fa.EdgeField[ta.wpfloat],
    tangent_y: fa.EdgeField[ta.wpfloat],
    tangent_z: fa.EdgeField[ta.wpfloat],
    normal_x: fa.EdgeField[ta.wpfloat],
    normal_y: fa.EdgeField[ta.wpfloat],
    normal_z: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_edge_tangent_and_normal(
        vertex_lat,
        vertex_lon,
        edge_lat,
        edge_lon,
        edge_orientation,
        out=(tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_component_of_edge_field_at_vertex(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute the zonal (u) an meridional (v) component of a cartesian vector (x, y, z) at the vertex position (lat, lon).

    The cartesian vector is defined on edges and it projection onto all 4 neighboring vertices of the diamond is computed.

    Args:
        vertex_lat: latitude of vertices
        vertex_lon: longitude of vertices
        x: x coordinate
        y: y coordinate
        z: z coordinate
    Returns:
        u_vertex_0: zonal (eastward positive) component at E2C2V[0]
        v_vertex_0: meridional (northward) component at E2C2V[0]
        u_vertex_1: zonal (eastward positive) component at E2C2V[1]
        v_vertex_1: meridional (northward) component at E2C2V[1]
        u_vertex_2: zonal (eastward positive) component at E2C2V[2]
        v_vertex_2: meridional (northward) component at E2C2V[2]
        u_vertex_3: zonal (eastward positive) component at E2C2V[3]
        v_vertex_3: meridional (northward) component at E2C2V[3]

    """
    vertex_lat_0 = vertex_lat(E2C2V[0])
    vertex_lon_0 = vertex_lon(E2C2V[0])
    u_vertex_0, v_vertex_0 = zonal_and_meridional_components_on_edges(
        vertex_lat_0, vertex_lon_0, x, y, z
    )
    vertex_lat_1 = vertex_lat(E2C2V[1])
    vertex_lon_1 = vertex_lon(E2C2V[1])
    u_vertex_1, v_vertex_1 = zonal_and_meridional_components_on_edges(
        vertex_lat_1, vertex_lon_1, x, y, z
    )
    vertex_lat_2 = vertex_lat(E2C2V[2])
    vertex_lon_2 = vertex_lon(E2C2V[2])
    u_vertex_2, v_vertex_2 = zonal_and_meridional_components_on_edges(
        vertex_lat_2, vertex_lon_2, x, y, z
    )
    vertex_lat_3 = vertex_lat(E2C2V[3])
    vertex_lon_3 = vertex_lon(E2C2V[3])
    u_vertex_3, v_vertex_3 = zonal_and_meridional_components_on_edges(
        vertex_lat_3, vertex_lon_3, x, y, z
    )
    return (
        u_vertex_0,
        v_vertex_0,
        u_vertex_1,
        v_vertex_1,
        u_vertex_2,
        v_vertex_2,
        u_vertex_3,
        v_vertex_3,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_component_of_edge_field_at_vertex(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
    u_vertex_1: fa.EdgeField[ta.wpfloat],
    v_vertex_1: fa.EdgeField[ta.wpfloat],
    u_vertex_2: fa.EdgeField[ta.wpfloat],
    v_vertex_2: fa.EdgeField[ta.wpfloat],
    u_vertex_3: fa.EdgeField[ta.wpfloat],
    v_vertex_3: fa.EdgeField[ta.wpfloat],
    u_vertex_4: fa.EdgeField[ta.wpfloat],
    v_vertex_4: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_component_of_edge_field_at_vertex(
        vertex_lat,
        vertex_lon,
        x,
        y,
        z,
        out=(
            u_vertex_1,
            v_vertex_1,
            u_vertex_2,
            v_vertex_2,
            u_vertex_3,
            v_vertex_3,
            u_vertex_4,
            v_vertex_4,
        ),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_component_of_edge_field_at_cell_center(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute zonal (U) and meridional (V) component of a vector (x, y, z) at cell centers (lat, lon)

    The vector is defined on edges and the projection is computed for the neighboring cell center s of the edge.

    Args:
        cell_lat: latitude of cell centers
        cell_lon: longitude of cell centers
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        u_cell_0: zonal (U) component at first cell neighbor of the edge E2C[0]
        v_cell_0: meridional (V) component at first cell neighbor of the edge E2C[1]
        u_cell_0: zonal (U) component at first cell neighbor of the edge E2C[0]
        v_cell_0: meridional (V) component at first cell neighbor of the edge E2C[1]

    """
    cell_lat_0 = cell_lat(E2C[0])
    cell_lon_0 = cell_lon(E2C[0])
    u_cell_0, v_cell_0 = zonal_and_meridional_components_on_edges(cell_lat_0, cell_lon_0, x, y, z)
    cell_lat_1 = cell_lat(E2C[1])
    cell_lon_1 = cell_lon(E2C[1])
    u_cell_1, v_cell_1 = zonal_and_meridional_components_on_edges(cell_lat_1, cell_lon_1, x, y, z)
    return (
        u_cell_0,
        v_cell_0,
        u_cell_1,
        v_cell_1,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_component_of_edge_field_at_cell_center(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
    u_cell_1: fa.EdgeField[ta.wpfloat],
    v_cell_1: fa.EdgeField[ta.wpfloat],
    u_cell_2: fa.EdgeField[ta.wpfloat],
    v_cell_2: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_component_of_edge_field_at_cell_center(
        cell_lat,
        cell_lon,
        x,
        y,
        z,
        out=(
            u_cell_1,
            v_cell_1,
            u_cell_2,
            v_cell_2,
        ),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def cell_center_arc_distance(
    lat_neighbor_0: fa.EdgeField[ta.wpfloat],
    lon_neighbor_0: fa.EdgeField[ta.wpfloat],
    lat_neighbor_1: fa.EdgeField[ta.wpfloat],
    lon_neighbor_1: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the distance between to cell centers.

    Computes the distance between the cell center of edge adjacent cells. This is a edge of the dual grid.

    Args:
        lat_neighbor_0: auxiliary vector of latitudes: cell centler of E2C[0] or edge center for boundary edges
        lon_neighbor_0: auxiliary vector of longitudes: cell centler of E2C[0] or edge center for boundary edges
        lat_neighbor_1: auxiliary vector of latitudes: cell centler of E2C[1] or edge center for boundary edges
        lon_neighbor_1: auxiliary vector of longitudes: cell centler of E2C[1] or edge center for boundary edges
        radius: radius of the sphere

    Returns:
        dual edge length

    """
    x0, y0, z0 = geographical_to_cartesian_on_edges(lat_neighbor_0, lon_neighbor_0)
    x1, y1, z1 = geographical_to_cartesian_on_edges(lat_neighbor_1, lon_neighbor_1)
    # (xi, yi, zi) are normalized by construction
    arc = arc_length(x0, x1, y0, y1, z0, z1, radius)
    return arc


@gtx.field_operator
def arc_distance_of_far_edges_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the arc length between the "far" vertices of an edge.

    Neighboring edges of an edge span up a diamond with 4 edges (E2C2E)  and 4 vertices (E2C2V). Here we compute the
    arc length between the two vertices in this diamond that are not directly connected to the edge:
    between d(v1, v3)
    v1-------v4
    |       /|
    |      / |
    |    e   |
    |  /     |
    |/       |
    v2 ------v3



    Args:
        vertex_lat: vertex latitude
        vertex_lon: vertex longitude
        radius: sphere radius

    Returns:
        arc length between the "far" vertices in the diamond.

    """
    x, y, z = geographical_to_cartesian_on_vertex(vertex_lat, vertex_lon)
    x2 = x(E2C2V[2])
    x3 = x(E2C2V[3])
    y2 = y(E2C2V[2])
    y3 = y(E2C2V[3])
    z2 = z(E2C2V[2])
    z3 = z(E2C2V[3])
    # (xi, yi, zi) are normalized by construction

    far_vertex_vertex_length = arc_length(x2, x3, y2, y3, z2, z3, radius)
    return far_vertex_vertex_length


@gtx.field_operator
def edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the arc length of an edge.

    This stencil could easily be inlined with `compute_arc_distance_of_far_edges_in_diamond`
    by using all indices in the E2C2V connectivity.
    They are kept separate due to different compute bounds.

    Args:
        vertex_lat: vertex latitudes
        vertex_lon: vertex longituds
        radius: sphere redius

    Returns:
        edge length
    """
    x, y, z = geographical_to_cartesian_on_vertex(vertex_lat, vertex_lon)
    x0 = x(E2V[0])
    x1 = x(E2V[1])
    y0 = y(E2V[0])
    y1 = y(E2V[1])
    z0 = z(E2V[0])
    z1 = z(E2V[1])
    # (xi, yi, zi) are normalized by construction

    length = arc_length(x0, x1, y0, y1, z0, z1, radius)
    return length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    edge_length(
        vertex_lat,
        vertex_lon,
        radius,
        out=length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cell_center_arc_distance(
    edge_neighbor_0_lat: fa.EdgeField[ta.wpfloat],
    edge_neighbor_0_lon: fa.EdgeField[ta.wpfloat],
    edge_neighbor_1_lat: fa.EdgeField[ta.wpfloat],
    edge_neighbor_1_lon: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cell_center_arc_distance(
        edge_neighbor_0_lat,
        edge_neighbor_0_lon,
        edge_neighbor_1_lat,
        edge_neighbor_1_lon,
        radius,
        out=dual_edge_length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_arc_distance_of_far_edges_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    far_vertex_distance: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    arc_distance_of_far_edges_in_diamond(
        vertex_lat,
        vertex_lon,
        radius,
        out=far_vertex_distance,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the area spanned by an edge and the its dual edge
    Args:
        owner_mask: owner mask for edges
        primal_edge_length: length of edge in primal grid
        dual_edge_length: length of edge in dual grid

    Returns:
        area

    """
    return where(owner_mask, primal_edge_length * dual_edge_length, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    area: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    edge_area(
        owner_mask,
        primal_edge_length,
        dual_edge_length,
        out=area,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the coriolis force on edges.
    Args:
       edge_center_lat: latitude of edge center
       angular_velocity: angular velocity

    Returns:
        coriolis parameter
    """
    return 2.0 * angular_velocity * sin(edge_center_lat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
    coriolis_parameter: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    coriolis_parameter_on_edges(
        edge_center_lat,
        angular_velocity,
        out=coriolis_parameter,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )
