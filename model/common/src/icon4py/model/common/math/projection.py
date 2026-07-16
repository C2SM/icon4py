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
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    """
    Compute gnomonic projection.

    gnomonic_proj
    Args:
        lon_c: longitude center on tangent plane
        lat_c: lattitude center on tangent plane
        lon: longitude point to be projected
        lat: lattitude point to be projected
    Returns:
        x, y: coordinates of projected point

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

    return x, y


def diff_on_edges_torus_numpy(
    *,
    cc_cv_x: data_alloc.NDArray,
    cc_cv_y: data_alloc.NDArray,
    cc_cell_x: data_alloc.NDArray,
    cc_cell_y: data_alloc.NDArray,
    domain_length: float,
    domain_height: float,
) -> tuple[data_alloc.NDArray, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(cc_cv_x)
    x_diff = cc_cell_x - cc_cv_x
    y_diff = cc_cell_y - cc_cv_y
    x_diff = array_ns.minimum(x_diff, domain_length - x_diff)
    y_diff = array_ns.minimum(y_diff, domain_height - y_diff)
    return x_diff, y_diff
