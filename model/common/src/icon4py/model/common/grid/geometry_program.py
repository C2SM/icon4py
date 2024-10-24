# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next import arccos, sin, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2V, E2V, EdgeDim
from icon4py.model.common.math.helpers import (
    cross_product,
    dot_product,
    normalize_cartesian_vector,
    spherical_to_cartesian_on_cells,
    spherical_to_cartesian_on_edges,
    spherical_to_cartesian_on_vertex,
    zonal_and_meridional_components_on_edges,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_of_edge_tangent(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Compute normalized cartesian vector tangential to an edge.

    That is the distance between the two vertices adjacent to the edge:
    t = d(v1, v2)
    """
    vertex_x, vertex_y, vertex_z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x = vertex_x(E2V[1]) - vertex_x(E2V[0])
    y = vertex_y(E2V[1]) - vertex_y(E2V[0])
    z = vertex_z(E2V[1]) - vertex_z(E2V[0])
    return x, y, z


@gtx.field_operator
def cartesian_coordinates_of_edge_normal(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_tangent_x: fa.EdgeField[ta.wpfloat],
    edge_tangent_y: fa.EdgeField[ta.wpfloat],
    edge_tangent_z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute the normal to the vector tangent.

    That is edge_center x |v1 - v2|, where v1 and v2 are the two vertices adjacent to an edge.
    """
    edge_center_x, edge_center_y, edge_center_z = spherical_to_cartesian_on_edges(
        edge_lat, edge_lon, r=1.0
    )
    cell_x, cell_y, cell_z = spherical_to_cartesian_on_cells(cell_lat, cell_lon, r=1.0)
    cell_distance_x = cell_x(E2C[1]) - cell_x(E2C[0])
    cell_distance_y = cell_y(E2C[1]) - cell_y(E2C[0])
    cell_distance_z = cell_z(E2C[1]) - cell_z(E2C[0])
    tangent_orientation_x, tangent_orientation_y, tangent_orientation_z = cross_product(
        edge_tangent_x,
        cell_distance_x,
        edge_tangent_y,
        cell_distance_y,
        edge_tangent_y,
        cell_distance_z,
    )
    projection = dot_product(
        edge_center_x,
        tangent_orientation_x,
        edge_center_y,
        tangent_orientation_y,
        edge_center_z,
        tangent_orientation_z,
    )
    tangent_orientation = where(projection >= 0.0, 1.0, -1.0)

    x, y, z = cross_product(
        edge_center_x, edge_tangent_x, edge_center_y, edge_tangent_y, edge_center_z, edge_tangent_z
    )
    normal_orientation = dot_product(cell_distance_x, x, cell_distance_y, y, cell_distance_z, z)
    x = where(normal_orientation < 0.0, -1.0 * x, x)
    y = where(normal_orientation < 0.0, -1.0 * y, y)
    z = where(normal_orientation < 0.0, -1.0 * z, z)
    edge_normal_x, edge_normal_y, edge_normal_z = normalize_cartesian_vector(x, y, z)
    return tangent_orientation, edge_normal_x, edge_normal_y, edge_normal_z


@gtx.field_operator
def cartesian_coordinates_edge_tangent_and_normal(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute normalized cartesian vectors of edge tangent and edge normal."""
    tangent_x, tangent_y, tangent_z = cartesian_coordinates_of_edge_tangent(vertex_lat, vertex_lon)
    tangent_orientation, normal_x, normal_y, normal_z = cartesian_coordinates_of_edge_normal(
        cell_lat, cell_lon, edge_lat, edge_lon, tangent_x, tangent_y, tangent_z
    )

    return tangent_orientation, tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_of_edge_tangent_and_normal(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
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
        cell_lat,
        cell_lon,
        vertex_lat,
        vertex_lon,
        edge_lat,
        edge_lon,
        out=(tangent_orientation, tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_primal_normal_vertex(
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
    """computes edges%primal_normal_cell, edges%primal_normal_vert"""
    vertex_lat_1 = vertex_lat(E2C2V[0])
    vertex_lon_1 = vertex_lon(E2C2V[0])
    u_vertex_1, v_vertex_1 = zonal_and_meridional_components_on_edges(
        vertex_lat_1, vertex_lon_1, x, y, z
    )
    vertex_lat_2 = vertex_lat(E2C2V[1])
    vertex_lon_2 = vertex_lon(E2C2V[1])
    u_vertex_2, v_vertex_2 = zonal_and_meridional_components_on_edges(
        vertex_lat_2, vertex_lon_2, x, y, z
    )
    vertex_lat_3 = vertex_lat(E2C2V[2])
    vertex_lon_3 = vertex_lon(E2C2V[2])
    u_vertex_3, v_vertex_3 = zonal_and_meridional_components_on_edges(
        vertex_lat_3, vertex_lon_3, x, y, z
    )
    vertex_lat_4 = vertex_lat(E2C2V[3])
    vertex_lon_4 = vertex_lon(E2C2V[3])
    u_vertex_4, v_vertex_4 = zonal_and_meridional_components_on_edges(
        vertex_lat_4, vertex_lon_4, x, y, z
    )
    return (
        u_vertex_1,
        v_vertex_1,
        u_vertex_2,
        v_vertex_2,
        u_vertex_3,
        v_vertex_3,
        u_vertex_4,
        v_vertex_4,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_primal_normal_vertex(
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
    edge_primal_normal_vertex(
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
def edge_primal_normal_cell(
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
    """computes edges%primal_normal_cell, edges%primal_normal_vert"""
    cell_lat_1 = cell_lat(E2C[0])
    cell_lon_1 = cell_lon(E2C[0])
    u_cell_1, v_cell_1 = zonal_and_meridional_components_on_edges(cell_lat_1, cell_lon_1, x, y, z)
    cell_lat_2 = cell_lat(E2C[1])
    cell_lon_2 = cell_lon(E2C[1])
    u_cell_2, v_cell_2 = zonal_and_meridional_components_on_edges(cell_lat_2, cell_lon_2, x, y, z)
    return u_cell_1, v_cell_1, u_cell_2, v_cell_2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_primal_normal_cell(
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
    edge_primal_normal_cell(
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


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cell_center_arc_distance(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Compute the length of dual edge.

    Distance between the cell center of edge adjacent cells. This is a edge of the dual grid and is
    orthogonal to the edge. dual_edge_length in ICON.
    """
    x, y, z = spherical_to_cartesian_on_cells(cell_lat, cell_lon, wpfloat(1.0))
    x0 = x(E2C[0])
    x1 = x(E2C[1])
    y0 = y(E2C[0])
    y1 = y(E2C[1])
    z0 = z(E2C[0])
    z1 = z(E2C[1])
    # (xi, yi, zi) are normalized by construction
    arc = radius * arccos(dot_product(x0, x1, y0, y1, z0, z1))
    return arc


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_arc_distance_of_far_edges_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Computes the length of a spherical edges
    - the direct edge length (primal_edge_length in ICON)ü
    - the length of the arc between the two far vertices in the diamond E2C2V (vertex_vertex_length in ICON)
    """
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x2 = x(E2C2V[2])
    x3 = x(E2C2V[3])
    y2 = y(E2C2V[2])
    y3 = y(E2C2V[3])
    z2 = z(E2C2V[2])
    z3 = z(E2C2V[3])
    # (xi, yi, zi) are normalized by construction

    far_vertex_vertex_length = radius * arccos(dot_product(x2, x3, y2, y3, z2, z3))
    return far_vertex_vertex_length


@gtx.field_operator
def compute_primal_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Computes the length of a spherical edges

    Called edge_length in ICON.
    The computation is the same as for the arc length between the far vertices in the E2C2V diamond but
    and could be done using the E2C2V connectivity, but is has different bounds, as there are no skip values for the edge adjacent vertices.
    """
    x, y, z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)
    x0 = x(E2V[0])
    x1 = x(E2V[1])
    y0 = y(E2V[0])
    y1 = y(E2V[1])
    z0 = z(E2V[0])
    z1 = z(E2V[1])
    # (xi, yi, zi) are normalized by construction

    edge_length = radius * arccos(dot_product(x0, x1, y0, y1, z0, z1))
    return edge_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    compute_primal_edge_length(
        vertex_lat,
        vertex_lon,
        radius,
        out=edge_length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _compute_dual_edge_length_and_far_vertex_distance_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    radius: ta.wpfloat,
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    far_vertex_distance = compute_arc_distance_of_far_edges_in_diamond(
        vertex_lat, vertex_lon, radius
    )
    dual_edge_length = cell_center_arc_distance(cell_lat, cell_lon, radius)
    return far_vertex_distance, dual_edge_length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_dual_edge_length_and_far_vertex_distance_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    radius: ta.wpfloat,
    far_vertex_distance: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _compute_dual_edge_length_and_far_vertex_distance_in_diamond(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        radius=radius,
        out=(far_vertex_distance, dual_edge_length),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """compute the edge_area"""
    return where(owner_mask, primal_edge_length * dual_edge_length, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
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
    """Compute the coriolis force on edges."""
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
