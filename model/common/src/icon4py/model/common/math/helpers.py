# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next import field_operator
from gt4py.next.ffront.fbuiltins import arccos, cos, sin, sqrt, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2V, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def average_cell_kdim_level_up(
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
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def average_edge_kdim_level_up(
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
    return 0.5 * (half_level_field + half_level_field(Koff[1]))


@field_operator
def difference_k_level_down(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels downwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field(Koff[-1]) - half_level_field


@field_operator
def difference_k_level_up(
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


@field_operator
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


@field_operator
def _grad_fd_tang(
    psi_v: gtx.Field[gtx.Dims[dims.VertexDim, dims.KDim], float],
    inv_primal_edge_length: fa.EdgeField[float],
    tangent_orientation: fa.EdgeField[float],
) -> fa.EdgeKField[float]:
    grad_tang_psi_e = tangent_orientation * (psi_v(E2V[1]) - psi_v(E2V[0])) * inv_primal_edge_length
    return grad_tang_psi_e


@gtx.field_operator
def geographical_to_cartesian_on_cells(
    lat: fa.CellField[ta.wpfloat], lon: fa.CellField[ta.wpfloat]
) -> tuple[fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat]]:
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
def geographical_to_cartesian_on_edges(
    lat: fa.EdgeField[ta.wpfloat], lon: fa.EdgeField[ta.wpfloat]
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
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
def geographical_to_cartesian_on_vertex(
    lat: fa.VertexField[ta.wpfloat], lon: fa.VertexField[ta.wpfloat]
) -> tuple[fa.VertexField[ta.wpfloat], fa.VertexField[ta.wpfloat], fa.VertexField[ta.wpfloat]]:
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
def dot_product(
    x1: fa.EdgeField[ta.wpfloat],
    x2: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    y2: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    z2: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def cross_product(
    x1: fa.EdgeField[ta.wpfloat],
    x2: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    y2: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    z2: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """Compute cross product of cartesian vectors (x1, y1, z1) x (x2, y2, z2)"""
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    return x, y, z


@gtx.field_operator
def norm2(
    x: fa.EdgeField[ta.wpfloat], y: fa.EdgeField[ta.wpfloat], z: fa.EdgeField[ta.wpfloat]
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product(x, x, y, y, z, z))


@gtx.field_operator
def normalize_cartesian_vector(
    v_x: fa.EdgeField[ta.wpfloat], v_y: fa.EdgeField[ta.wpfloat], v_z: fa.EdgeField[ta.wpfloat]
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Normalize a cartesian vector.

    Args:
        v_x: x coordinate
        v_y: y coordinate
        v_z: z coordinate

    Returns:
        normalized vector

    """
    norm = norm2(v_x, v_y, v_z)
    return v_x / norm, v_y / norm, v_z / norm


@gtx.field_operator
def invert(f: fa.EdgeField[ta.wpfloat]) -> fa.EdgeField[ta.wpfloat]:
    """
    Invert values.
    Args:
        f: values

    Returns:
        1/f where f is not zero.
    """
    return where(f != 0.0, 1.0 / f, f)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_inverse(
    f: fa.EdgeField[ta.wpfloat],
    f_inverse: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    invert(f, out=f_inverse, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_components_on_cells(
    lat: fa.CellField[ta.wpfloat],
    lon: fa.CellField[ta.wpfloat],
    x: fa.CellField[ta.wpfloat],
    y: fa.CellField[ta.wpfloat],
    z: fa.CellField[ta.wpfloat],
) -> tuple[fa.CellField[ta.wpfloat], fa.CellField[ta.wpfloat]]:
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
    lat: fa.EdgeField[ta.wpfloat],
    lon: fa.EdgeField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
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
    lat: fa.EdgeField[ta.wpfloat],
    lon: fa.EdgeField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
    u: fa.EdgeField[ta.wpfloat],
    v: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_components_on_edges(
        lat, lon, x, y, z, out=(u, v), domain={dims.EdgeDim: (horizontal_start, horizontal_end)}
    )


@gtx.field_operator
def cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[ta.wpfloat],
    lon: fa.EdgeField[ta.wpfloat],
    u: fa.EdgeField[ta.wpfloat],
    v: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
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

    norm = norm2(x, y, z)
    return x / norm, y / norm, y / norm


@gtx.program
def compute_cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    u: fa.EdgeField[ta.wpfloat],
    v: fa.EdgeField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
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


@gtx.field_operator
def arc_length(
    x0: fa.EdgeField[ta.wpfloat],
    x1: fa.EdgeField[ta.wpfloat],
    y0: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    z0: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
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
    return radius * arccos(dot_product(x0, x1, y0, y1, z0, z1))
