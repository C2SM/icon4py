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

from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.grid.horizontal import HorizontalMarkerIndex
from icon4py.model.common.grid.icon import IconGrid


def zonalwind_2_normalwind_jabw_numpy(
    icon_grid: IconGrid,
    jw_u0: float,
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: np.array,
    edge_lon: np.array,
    primal_normal_x: np.array,
    eta_v_e: np.array,
):
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_e).

    Args:
        icon_grid: IconGrid
        jw_u0: base zonal wind speed factor
        jw_up: perturbation amplitude
        lat_perturbation_center: perturbation center in latitude
        lon_perturbation_center: perturbation center in longitude
        edge_lat: edge center latitude
        edge_lon: edge center longitude
        primal_normal_x: zonal component of primal normal vector at edge center
        eta_v_e: vertical eta coordinate at edge center
    Returns: normal wind
    """
    mask = np.ones((icon_grid.num_edges, icon_grid.num_levels), dtype=bool)
    mask[
        0 : icon_grid.get_end_index(EdgeDim, HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1), :
    ] = False
    edge_lat = np.repeat(np.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = np.repeat(np.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = np.repeat(np.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1)
    u = np.where(mask, jw_u0 * (np.cos(eta_v_e) ** 1.5) * (np.sin(2.0 * edge_lat) ** 2), 0.0)
    if jw_up > 1.0e-20:
        u = np.where(
            mask,
            u
            + jw_up
            * np.exp(
                -10.0
                * np.arccos(
                    np.sin(lat_perturbation_center) * np.sin(edge_lat)
                    + np.cos(lat_perturbation_center)
                    * np.cos(edge_lat)
                    * np.cos(edge_lon - lon_perturbation_center)
                )
                ** 2
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn
