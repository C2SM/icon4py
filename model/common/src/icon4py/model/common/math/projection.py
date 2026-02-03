# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np

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
    cosc = np.sin(lat_c) * np.sin(lat) + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c)
    zk = 1.0 / cosc

    x = zk * np.cos(lat) * np.sin(lon - lon_c)
    y = zk * (np.cos(lat_c) * np.sin(lat) - np.sin(lat_c) * np.cos(lat) * np.cos(lon - lon_c))

    return x, y


def gnomonic_proj_single_val(lon_c: float, lat_c: float, lon: float, lat: float) -> tuple:
    cosc = math.sin(lat_c) * math.sin(lat) + math.cos(lat_c) * math.cos(lat) * math.cos(lon - lon_c)
    zk = 1.0 / cosc

    x = zk * math.cos(lat) * math.sin(lon - lon_c)
    y = zk * (
        math.cos(lat_c) * math.sin(lat) - math.sin(lat_c) * math.cos(lat) * math.cos(lon - lon_c)
    )
    return x, y


def plane_torus_closest_coordinates(
    cc_cv_x: tuple, cc_cell_x: float, cc_cell_y: float, domain_length: float, domain_height: float
) -> tuple:
    x0 = cc_cv_x[0]
    x1 = cc_cell_x
    y0 = cc_cv_x[1]
    y1 = cc_cell_y

    x1 = np.where(
        abs(x1 - x0) <= 0.5 * domain_length,
        x1,
        np.where(x0 > x1, x1 + domain_length, x1 - domain_length),
    )
    y1 = np.where(
        abs(y1 - y0) <= 0.5 * domain_height,
        y1,
        np.where(y0 > y1, y1 + domain_height, y1 - domain_height),
    )
    return x1, y1
