# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


def gnomonic_proj(
    lon_c: np.ndarray,
    lat_c: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """
    Compute gnomonic projection.

    gnomonic_proj
    Args:
        lon_c, lat_c: center on tangent plane
        lat, lon: point to be projected
    Return values:
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
