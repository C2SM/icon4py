# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next import arccos, cos, sin, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2V, Koff
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def average_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return wpfloat(0.5) * (half_level_field + half_level_field(Koff[1]))


@gtx.field_operator
def average_level_plus1_on_edges(
    half_level_field: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over an edge field for storage
    in the corresponding full levels.
    Args:
        half_level_field: fa.EdgeKField[wpfloat]

    Returns: fa.EdgeKField[wpfloat] full level field

    """
    return wpfloat(0.5) * (half_level_field + half_level_field(Koff[1]))


@gtx.field_operator
def difference_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field - half_level_field(Koff[1])


@gtx.field_operator
def grad_fd_norm(
    psi_c: fa.CellKField[float],
    inv_dual_edge_length: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    """
    Calculate the gradient value of adjacent interface levels.

    Computes the difference of two offseted values multiplied by a field of the offseted dimension
    Args:
        psi_c: fa.CellKField[float],
        inv_dual_edge_length: Field[Dims[EdgeDim], float],

    Returns: fa.EdgeKField[float]

    """
    grad_norm_psi_e = (psi_c(E2C[1]) - psi_c(E2C[0])) * inv_dual_edge_length
    return grad_norm_psi_e


@gtx.field_operator
def _grad_fd_tang(
    psi_v: gtx.Field[gtx.Dims[dims.VertexDim, dims.KDim], float],
    inv_primal_edge_length: fa.EdgeField[float],
    tangent_orientation: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def geographical_to_cartesian_on_cells(
    lat: fa.CellField[wpfloat], lon: fa.CellField[wpfloat]
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Convert geographical (lat, lon) coordinates to cartesian coordinates on the unit sphere.

    Args:
        lat: latitude
        lon: longitude

    Returns:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    """
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    return x, y, z


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def geographical_to_cartesian_on_edges(
    lat: fa.EdgeField[wpfloat], lon: fa.EdgeField[wpfloat]
) -> tuple[fa.EdgeField[wpfloat], fa.EdgeField[wpfloat], fa.EdgeField[wpfloat]]:
    """
    Convert geographical (lat, lon) coordinates to cartesian coordinates on the unit sphere.

    Args:
        lat: latitude
        lon: longitude

    Returns:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    """
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    return x, y, z


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def geographical_to_cartesian_on_vertices(
    lat: fa.VertexField[wpfloat], lon: fa.VertexField[wpfloat]
) -> tuple[fa.VertexField[wpfloat], fa.VertexField[wpfloat], fa.VertexField[wpfloat]]:
    """
    Convert geographical (lat, lon) coordinates to cartesian coordinates on the unit sphere.

    Args:
        lat: latitude
        lon: longitude

    Returns:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    """
    x = cos(lat) * cos(lon)
    y = cos(lat) * sin(lon)
    z = sin(lat)
    return x, y, z


@gtx.field_operator
def dot_product_on_edges(
    x1: fa.EdgeField[wpfloat],
    x2: fa.EdgeField[wpfloat],
    y1: fa.EdgeField[wpfloat],
    y2: fa.EdgeField[wpfloat],
    z1: fa.EdgeField[wpfloat],
    z2: fa.EdgeField[wpfloat],
) -> fa.EdgeField[wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_cells(
    x1: fa.CellField[wpfloat],
    x2: fa.CellField[wpfloat],
    y1: fa.CellField[wpfloat],
    y2: fa.CellField[wpfloat],
    z1: fa.CellField[wpfloat],
    z2: fa.CellField[wpfloat],
) -> fa.CellField[wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_vertices(
    x1: fa.VertexField[wpfloat],
    x2: fa.VertexField[wpfloat],
    y1: fa.VertexField[wpfloat],
    y2: fa.VertexField[wpfloat],
    z1: fa.VertexField[wpfloat],
    z2: fa.VertexField[wpfloat],
) -> fa.VertexField[wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def cross_product_on_edges(
    x1: fa.EdgeField[wpfloat],
    x2: fa.EdgeField[wpfloat],
    y1: fa.EdgeField[wpfloat],
    y2: fa.EdgeField[wpfloat],
    z1: fa.EdgeField[wpfloat],
    z2: fa.EdgeField[wpfloat],
) -> tuple[fa.EdgeField[wpfloat], fa.EdgeField[wpfloat], fa.EdgeField[wpfloat]]:
    """Compute cross product of cartesian vectors (x1, y1, z1) x (x2, y2, z2)"""
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    return x, y, z


@gtx.field_operator
def norm2_on_edges(
    x: fa.EdgeField[wpfloat], y: fa.EdgeField[wpfloat], z: fa.EdgeField[wpfloat]
) -> fa.EdgeField[wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_edges(x, x, y, y, z, z))


@gtx.field_operator
def norm2_on_cells(
    x: fa.CellField[wpfloat], y: fa.CellField[wpfloat], z: fa.CellField[wpfloat]
) -> fa.CellField[wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_cells(x, x, y, y, z, z))


@gtx.field_operator
def norm2_on_vertices(
    x: fa.VertexField[wpfloat], y: fa.VertexField[wpfloat], z: fa.VertexField[wpfloat]
) -> fa.VertexField[wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_vertices(x, x, y, y, z, z))


@gtx.field_operator
def normalize_cartesian_vector_on_edges(
    v_x: fa.EdgeField[wpfloat], v_y: fa.EdgeField[wpfloat], v_z: fa.EdgeField[wpfloat]
) -> tuple[fa.EdgeField[wpfloat], fa.EdgeField[wpfloat], fa.EdgeField[wpfloat]]:
    """
    Normalize a cartesian vector.

    Args:
        v_x: x coordinate
        v_y: y coordinate
        v_z: z coordinate

    Returns:
        normalized vector

    """
    norm = norm2_on_edges(v_x, v_y, v_z)
    return v_x / norm, v_y / norm, v_z / norm


@gtx.field_operator
def invert_edge_field(f: fa.EdgeField[wpfloat]) -> fa.EdgeField[wpfloat]:
    """
    Invert values.
    Args:
        f: values

    Returns:
        1/f where f is not zero.
    """
    return where(f != wpfloat(0.0), wpfloat(1.0) / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse_on_edges(
    f: fa.EdgeField[wpfloat],
    f_inverse: fa.EdgeField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    invert_edge_field(f, out=f_inverse, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_components_on_cells(
    lat: fa.CellField[wpfloat],
    lon: fa.CellField[wpfloat],
    x: fa.CellField[wpfloat],
    y: fa.CellField[wpfloat],
    z: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute normalized zonal and meridional components of a cartesian vector (x, y, z) at point (lat, lon)

    Args:
        lat: latitude
        lon: longitude
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        u zonal component
        v meridional component

    """
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)
    u = cos_lon * y - sin_lon * x

    v = cos_lat * z - sin_lat * (cos_lon * x + sin_lon * y)
    norm = sqrt(u * u + v * v)
    return u / norm, v / norm


@gtx.field_operator
def zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[wpfloat],
    lon: fa.EdgeField[wpfloat],
    x: fa.EdgeField[wpfloat],
    y: fa.EdgeField[wpfloat],
    z: fa.EdgeField[wpfloat],
) -> tuple[fa.EdgeField[wpfloat], fa.EdgeField[wpfloat]]:
    """
    Compute the zonal and meridional component of a vector (x, y, z) at position (lat, lon)

    Args:
        lat: latitude
        lon: longitude
        x: x component of cartesian vector
        y: y component of cartesian vector
        z: z component of cartesian vector

    Returns:
        zonal (eastward) component of (x, y, z) at (lat, lon)
        meridional (northward) component of (x, y, z) at (lat, lon)

    """
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)
    u = cos_lon * y - sin_lon * x

    v = cos_lat * z - sin_lat * (cos_lon * x + sin_lon * y)
    norm = sqrt(u * u + v * v)
    return u / norm, v / norm


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[wpfloat],
    lon: fa.EdgeField[wpfloat],
    x: fa.EdgeField[wpfloat],
    y: fa.EdgeField[wpfloat],
    z: fa.EdgeField[wpfloat],
    u: fa.EdgeField[wpfloat],
    v: fa.EdgeField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_components_on_edges(
        lat, lon, x, y, z, out=(u, v), domain={dims.EdgeDim: (horizontal_start, horizontal_end)}
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[wpfloat],
    lon: fa.EdgeField[wpfloat],
    u: fa.EdgeField[wpfloat],
    v: fa.EdgeField[wpfloat],
) -> tuple[fa.EdgeField[wpfloat], fa.EdgeField[wpfloat], fa.EdgeField[wpfloat]]:
    """
    Compute cartesian coordinates from zonal an meridional components at position (lat, lon)
    Args:
        lat: latitude
        lon: longitude
        u: zonal component
        v: meridional component

    Returns:
        x, y, z cartesian components

    """
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)

    x = -u * sin_lon - v * sin_lat * cos_lon
    y = u * cos_lon - v * sin_lat * sin_lon
    z = cos_lat * v

    norm = norm2_on_edges(x, y, z)
    return x / norm, y / norm, z / norm


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
    edge_lat: fa.EdgeField[wpfloat],
    edge_lon: fa.EdgeField[wpfloat],
    u: fa.EdgeField[wpfloat],
    v: fa.EdgeField[wpfloat],
    x: fa.EdgeField[wpfloat],
    y: fa.EdgeField[wpfloat],
    z: fa.EdgeField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
        edge_lat,
        edge_lon,
        u,
        v,
        out=(x, y, z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_from_zonal_and_meridional_components_on_cells(
    lat: fa.CellField[wpfloat],
    lon: fa.CellField[wpfloat],
    u: fa.CellField[wpfloat],
    v: fa.CellField[wpfloat],
) -> tuple[fa.CellField[wpfloat], fa.CellField[wpfloat], fa.CellField[wpfloat]]:
    """
    Compute cartesian coordinates form zonal an meridonal components at position (lat, lon)
    Args:
        lat: latitude
        lon: longitude
        u: zonal component
        v: meridional component

    Returns:
        x, y, z cartesian components

    """
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)

    x = -u * sin_lon - v * sin_lat * cos_lon
    y = u * cos_lon - v * sin_lat * sin_lon
    z = cos_lat * v

    norm = norm2_on_cells(x, y, z)
    return x / norm, y / norm, z / norm


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_from_zonal_and_meridional_components_on_cells(
    cell_lat: fa.CellField[wpfloat],
    cell_lon: fa.CellField[wpfloat],
    u: fa.CellField[wpfloat],
    v: fa.CellField[wpfloat],
    x: fa.CellField[wpfloat],
    y: fa.CellField[wpfloat],
    z: fa.CellField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_from_zonal_and_meridional_components_on_cells(
        cell_lat,
        cell_lon,
        u,
        v,
        out=(x, y, z),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def arc_length_on_edges(
    x0: fa.EdgeField[wpfloat],
    x1: fa.EdgeField[wpfloat],
    y0: fa.EdgeField[wpfloat],
    y1: fa.EdgeField[wpfloat],
    z0: fa.EdgeField[wpfloat],
    z1: fa.EdgeField[wpfloat],
    radius: wpfloat,
):
    """
    Compute the arc length between two points on the sphere.

    Inputs are cartesian coordinates of the points.

    Args:
        x0: x coordinate of point_0
        x1: x coordinate of point_1
        y0: y coordinate of point_0
        y1: y coordinate of point_1
        z0: z coordinate of point_0
        z1: z coordinate of point_1
        radius: sphere radius

    Returns:
        arc length

    """
    return radius * arccos(dot_product_on_edges(x0, x1, y0, y1, z0, z1))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_edges(
    input_field: fa.EdgeKField[wpfloat],
    average: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    average_level_plus1_on_edges(
        input_field,
        out=average,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_cells(
    input_field: fa.CellKField[wpfloat],
    average: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    average_level_plus1_on_cells(
        input_field,
        out=average,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
