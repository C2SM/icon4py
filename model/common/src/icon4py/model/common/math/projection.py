# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

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
