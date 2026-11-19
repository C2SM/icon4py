# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Coordinate transformation operations on unstructured grid fields.

Contains conversions between geographical (lat/lon) and cartesian coordinates,
and between zonal/meridional and cartesian components.
"""

from gt4py import next as gtx
from gt4py.next import cos, sin, sqrt

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vector_operations import norm2_on_cells, norm2_on_edges


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def geographical_to_cartesian_on_cells(
    lat: fa.CellField[gtx.float64], lon: fa.CellField[gtx.float64]
) -> tuple[fa.CellField[gtx.float64], fa.CellField[gtx.float64], fa.CellField[gtx.float64]]:
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
    lat: fa.EdgeField[gtx.float64], lon: fa.EdgeField[gtx.float64]
) -> tuple[fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64]]:
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
    lat: fa.VertexField[gtx.float64], lon: fa.VertexField[gtx.float64]
) -> tuple[fa.VertexField[gtx.float64], fa.VertexField[gtx.float64], fa.VertexField[gtx.float64]]:
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
def zonal_and_meridional_components_on_cells(
    lat: fa.CellField[gtx.float64],
    lon: fa.CellField[gtx.float64],
    x: fa.CellField[gtx.float64],
    y: fa.CellField[gtx.float64],
    z: fa.CellField[gtx.float64],
) -> tuple[fa.CellField[gtx.float64], fa.CellField[gtx.float64]]:
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
    lat: fa.EdgeField[gtx.float64],
    lon: fa.EdgeField[gtx.float64],
    x: fa.EdgeField[gtx.float64],
    y: fa.EdgeField[gtx.float64],
    z: fa.EdgeField[gtx.float64],
) -> tuple[fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64]]:
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
def compute_zonal_and_meridional_components_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    lat: fa.EdgeField[gtx.float64],
    lon: fa.EdgeField[gtx.float64],
    x: fa.EdgeField[gtx.float64],
    y: fa.EdgeField[gtx.float64],
    z: fa.EdgeField[gtx.float64],
    u: fa.EdgeField[gtx.float64],
    v: fa.EdgeField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_components_on_edges(
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        z=z,
        out=(u, v),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
    lat: fa.EdgeField[gtx.float64],
    lon: fa.EdgeField[gtx.float64],
    u: fa.EdgeField[gtx.float64],
    v: fa.EdgeField[gtx.float64],
) -> tuple[fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64], fa.EdgeField[gtx.float64]]:
    """
    Compute cartesian coordinates from zonal and meridional components at position (lat, lon)
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
def compute_cartesian_coordinates_from_zonal_and_meridional_components_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    edge_lat: fa.EdgeField[gtx.float64],
    edge_lon: fa.EdgeField[gtx.float64],
    u: fa.EdgeField[gtx.float64],
    v: fa.EdgeField[gtx.float64],
    x: fa.EdgeField[gtx.float64],
    y: fa.EdgeField[gtx.float64],
    z: fa.EdgeField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_from_zonal_and_meridional_components_on_edges(
        lat=edge_lat,
        lon=edge_lon,
        u=u,
        v=v,
        out=(x, y, z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_from_zonal_and_meridional_components_on_cells(
    lat: fa.CellField[gtx.float64],
    lon: fa.CellField[gtx.float64],
    u: fa.CellField[gtx.float64],
    v: fa.CellField[gtx.float64],
) -> tuple[fa.CellField[gtx.float64], fa.CellField[gtx.float64], fa.CellField[gtx.float64]]:
    """
    Compute cartesian coordinates from zonal and meridional components at position (lat, lon)
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
def compute_cartesian_coordinates_from_zonal_and_meridional_components_on_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_lat: fa.CellField[gtx.float64],
    cell_lon: fa.CellField[gtx.float64],
    u: fa.CellField[gtx.float64],
    v: fa.CellField[gtx.float64],
    x: fa.CellField[gtx.float64],
    y: fa.CellField[gtx.float64],
    z: fa.CellField[gtx.float64],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_from_zonal_and_meridional_components_on_cells(
        lat=cell_lat,
        lon=cell_lon,
        u=u,
        v=v,
        out=(x, y, z),
        domain={dims.CellDim: (horizontal_start, horizontal_end)},
    )
