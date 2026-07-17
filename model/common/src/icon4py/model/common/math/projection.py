# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from icon4py.model.common.utils import data_allocation as data_alloc


def gnomonic_proj(
    lon_c: data_alloc.NDArray,
    lat_c: data_alloc.NDArray,
    lon: data_alloc.NDArray,
    lat: data_alloc.NDArray,
    sphere_radius: float,
) -> data_alloc.NDArray:
    """
    Compute gnomonic projection onto a tangent plane with origin at (lon_c, lat_c).

    gnomonic_proj
    Args:
        lon_c: longitude center on tangent plane
        lat_c: lattitude center on tangent plane
        lon: longitude point to be projected
        lat: lattitude point to be projected
        sphere_radius: radius of the sphere
    Returns:
        x, y: x and y coordinates of the projected point on the tangent plane

    Variables:
        zk: scale factor perpendicular to the radius from the center of the map
        cosc: cosine of the angular distance of the given point (lat,lon) from the center of projection
    LITERATURE:
        Map Projections: A Working Manual, Snyder, 1987, p. 165
    TODO:
        replace this with a suitable library call
    """
    array_ns = data_alloc.array_namespace(lon_c)
    cosc = array_ns.sin(lat_c) * array_ns.sin(lat) + array_ns.cos(lat_c) * array_ns.cos(
        lat
    ) * array_ns.cos(lon - lon_c)
    zk = 1.0 / cosc

    x = zk * array_ns.cos(lat) * array_ns.sin(lon - lon_c)
    y = zk * (
        array_ns.cos(lat_c) * array_ns.sin(lat)
        - array_ns.sin(lat_c) * array_ns.cos(lat) * array_ns.cos(lon - lon_c)
    )
    return array_ns.column_stack((x, y)) * sphere_radius


def compute_cell_distance_on_torus(
    *,
    source_cell_x: data_alloc.NDArray,
    source_cell_y: data_alloc.NDArray,
    target_cell_x: data_alloc.NDArray,
    target_cell_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(source_cell_x)
    x_diff = target_cell_x - source_cell_x
    y_diff = target_cell_y - source_cell_y
    x_diff = array_ns.where(
        domain_length - array_ns.abs(x_diff) > array_ns.abs(x_diff),
        x_diff,
        x_diff - array_ns.sign(x_diff) * domain_length,
    )
    y_diff = array_ns.where(
        domain_height - array_ns.abs(y_diff) > array_ns.abs(y_diff),
        y_diff,
        y_diff - array_ns.sign(y_diff) * domain_height,
    )
    return array_ns.column_stack((x_diff, y_diff))
