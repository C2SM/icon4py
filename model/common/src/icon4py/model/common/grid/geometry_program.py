from gt4py import next as gtx
from gt4py.next import arccos, neighbor_sum, sin, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2V, E2V, E2VDim, EdgeDim
from icon4py.model.common.math.helpers import (
    compute_zonal_and_meridional_components_on_edges,
    cross_product,
    dot_product,
    normalize_cartesian_vector,
    spherical_to_cartesian_on_cells,
    spherical_to_cartesian_on_edges,
    spherical_to_cartesian_on_vertex,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_primal_tangent(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Compute normalized cartesian vector tangential to an edge.

    That is the normalized distance between the two vertices adjacent to the edge:
    t = |v1 - v2|
    """
    vertex_x, vertex_y, vertex_z = spherical_to_cartesian_on_vertex(vertex_lat, vertex_lon, 1.0)

    x = neighbor_sum(subtract_coeff * vertex_x(E2V), axis=E2VDim)
    y = neighbor_sum(subtract_coeff * vertex_y(E2V), axis=E2VDim)
    z = neighbor_sum(subtract_coeff * vertex_z(E2V), axis=E2VDim)
    return normalize_cartesian_vector(x, y, z)


@gtx.field_operator
def edge_primal_normal(
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_tangent_x: fa.EdgeField[ta.wpfloat],
    edge_tangent_y: fa.EdgeField[ta.wpfloat],
    edge_tangent_z: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """Compute the normal to the vector tangent.

    That is edge_center x |v1 - v2|, where v1 and v2 are the two vertices adjacent to an edge.
    """
    edge_center_x, edge_center_y, edge_center_z = spherical_to_cartesian_on_edges(
        edge_lat, edge_lon, r=1.0
    )
    x, y, z = cross_product(
        edge_center_x, edge_tangent_x, edge_center_y, edge_tangent_y, edge_center_z, edge_tangent_z
    )
    return normalize_cartesian_vector(x, y, z)


@gtx.field_operator
def edge_tangent_and_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute normalized cartesian vectors of edge tangent and edge normal."""
    tangent_x, tangent_y, tangent_z = edge_primal_tangent(vertex_lat, vertex_lon, subtract_coeff)
    normal_x, normal_y, normal_z = edge_primal_normal(
        edge_lat, edge_lon, tangent_x, tangent_z, tangent_z
    )
    return tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_tangent_and_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    subtract_coeff: gtx.Field[gtx.Dims[EdgeDim, E2VDim], ta.wpfloat],
    tangent_x: fa.EdgeField[ta.wpfloat],
    tangent_y: fa.EdgeField[ta.wpfloat],
    tangent_z: fa.EdgeField[ta.wpfloat],
    normal_x: fa.EdgeField[ta.wpfloat],
    normal_y: fa.EdgeField[ta.wpfloat],
    normal_z: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    edge_tangent_and_normal(
        vertex_lat,
        vertex_lon,
        edge_lat,
        edge_lon,
        subtract_coeff,
        out=(tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_primal_normals_cell_vert(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
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
    cell_lat_1 = cell_lat(E2C[0])
    cell_lon_1 = cell_lon(E2C[0])
    u_cell_1, v_cell_1 = compute_zonal_and_meridional_components_on_edges(
        cell_lat_1, cell_lon_1, x, y, z
    )
    cell_lat_2 = cell_lat(E2C[1])
    cell_lon_2 = cell_lon(E2C[1])
    u_cell_2, v_cell_2 = compute_zonal_and_meridional_components_on_edges(
        cell_lat_2, cell_lon_2, x, y, z
    )
    vertex_lat_1 = vertex_lat(E2V[0])
    vertex_lon_1 = vertex_lon(E2V[0])
    u_vertex_1, v_vertex_1 = compute_zonal_and_meridional_components_on_edges(
        vertex_lat_1, vertex_lon_1, x, y, z
    )
    vertex_lat_2 = vertex_lat(E2V[1])
    vertex_lon_2 = vertex_lon(E2V[1])
    u_vertex_2, v_vertex_2 = compute_zonal_and_meridional_components_on_edges(
        vertex_lat_2, vertex_lon_2, x, y, z
    )
    return u_cell_1, v_cell_1, u_cell_2, v_cell_2, u_vertex_1, v_vertex_1, u_vertex_2, v_vertex_2


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
def _edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """compute the edge_area"""
    return where(owner_mask, primal_edge_length * dual_edge_length, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[fa.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    edge_area: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _edge_area(
        owner_mask,
        primal_edge_length,
        dual_edge_length,
        out=edge_area,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def _coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """Compute the coriolis force on edges."""
    return 2.0 * angular_velocity * sin(edge_center_lat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
    coriolis_parameter: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    _coriolis_parameter_on_edges(
        edge_center_lat,
        angular_velocity,
        out=coriolis_parameter,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )
