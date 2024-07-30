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

from icon4py.model.common import constants as phy_const
from icon4py.model.common.dimension import EdgeDim
from icon4py.model.common.grid import horizontal as h_grid, icon as icon_grid
from icon4py.model.common.settings import xp


def hydrostatic_adjustment_numpy(
    wgtfac_c: xp.ndarray,
    ddqz_z_half: xp.ndarray,
    exner_ref_mc: xp.ndarray,
    d_exner_dz_ref_ic: xp.ndarray,
    theta_ref_mc: xp.ndarray,
    theta_ref_ic: xp.ndarray,
    rho: xp.ndarray,
    exner: xp.ndarray,
    theta_v: xp.ndarray,
    num_levels: int,
):
    # virtual temperature
    temp_v = theta_v * exner

    for k in range(num_levels - 2, -1, -1):
        fac1 = (
            wgtfac_c[:, k + 1] * (temp_v[:, k + 1] - theta_ref_mc[:, k + 1] * exner[:, k + 1])
            - (1.0 - wgtfac_c[:, k + 1]) * theta_ref_mc[:, k] * exner[:, k + 1]
        )
        fac2 = (1.0 - wgtfac_c[:, k + 1]) * temp_v[:, k] * exner[:, k + 1]
        fac3 = exner_ref_mc[:, k + 1] - exner_ref_mc[:, k] - exner[:, k + 1]

        quadratic_a = (theta_ref_ic[:, k + 1] * exner[:, k + 1] + fac1) / ddqz_z_half[:, k + 1]
        quadratic_b = -(
            quadratic_a * fac3 + fac2 / ddqz_z_half[:, k + 1] + fac1 * d_exner_dz_ref_ic[:, k + 1]
        )
        quadratic_c = -(fac2 * fac3 / ddqz_z_half[:, k + 1] + fac2 * d_exner_dz_ref_ic[:, k + 1])

        exner[:, k] = (quadratic_b + xp.sqrt(quadratic_b**2 + 4.0 * quadratic_a * quadratic_c)) / (
            2.0 * quadratic_a
        )
        theta_v[:, k] = temp_v[:, k] / exner[:, k]
        rho[:, k] = exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])

    return rho, exner, theta_v


def hydrostatic_adjustment_constant_thetav_numpy(
    wgtfac_c: xp.ndarray,
    ddqz_z_half: xp.ndarray,
    exner_ref_mc: xp.ndarray,
    d_exner_dz_ref_ic: xp.ndarray,
    theta_ref_mc: xp.ndarray,
    theta_ref_ic: xp.ndarray,
    rho: xp.ndarray,
    exner: xp.ndarray,
    theta_v: xp.ndarray,
    num_levels: int,
) -> tuple[xp.ndarray, xp.ndarray]:
    """
    Computes a hydrostatically balanced profile. In constrast to the above
    hydrostatic_adjustment_numpy, the virtual temperature is kept (assumed)
    constant during the adjustment, leading to a simpler formula.
    """

    for k in range(num_levels - 2, -1, -1):
        theta_v_pr_ic = wgtfac_c[:, k + 1] * (theta_v[:, k + 1] - theta_ref_mc[:, k + 1]) + (
            1.0 - wgtfac_c[:, k + 1]
        ) * (theta_v[:, k] - theta_ref_mc[:, k])

        exner[:, k] = (
            exner[:, k + 1]
            + (exner_ref_mc[:, k] - exner_ref_mc[:, k + 1])
            - ddqz_z_half[:, k + 1]
            / (theta_v_pr_ic + theta_ref_ic[:, k + 1])
            * theta_v_pr_ic
            * d_exner_dz_ref_ic[:, k + 1]
        )

    for k in range(num_levels - 1, -1, -1):
        rho[:, k] = exner[:, k] ** phy_const.CVD_O_RD * phy_const.P0REF / (phy_const.RD * theta_v[:, k])

    return rho, exner


def zonalwind_2_normalwind_numpy(
    grid: icon_grid.IconGrid,
    jw_u0: float,
    jw_up: float,
    lat_perturbation_center: float,
    lon_perturbation_center: float,
    edge_lat: xp.ndarray,
    edge_lon: xp.ndarray,
    primal_normal_x: xp.ndarray,
    eta_v_e: xp.ndarray,
):
    """
    Compute normal wind at edge center from vertical eta coordinate (eta_v_e).

    Args:
        grid: IconGrid
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
    # TODO (Chia Rui) this function needs a test

    mask = xp.ones((grid.num_edges, grid.num_levels), dtype=bool)
    mask[
        0 : grid.get_end_index(EdgeDim, h_grid.HorizontalMarkerIndex.lateral_boundary(EdgeDim) + 1), :
    ] = False
    edge_lat = xp.repeat(xp.expand_dims(edge_lat, axis=-1), eta_v_e.shape[1], axis=1)
    edge_lon = xp.repeat(xp.expand_dims(edge_lon, axis=-1), eta_v_e.shape[1], axis=1)
    primal_normal_x = xp.repeat(xp.expand_dims(primal_normal_x, axis=-1), eta_v_e.shape[1], axis=1)
    u = xp.where(mask, jw_u0 * (xp.cos(eta_v_e) ** 1.5) * (xp.sin(2.0 * edge_lat) ** 2), 0.0)
    if jw_up > 1.0e-20:
        u = xp.where(
            mask,
            u
            + jw_up
            * xp.exp(
                -10.0
                * xp.arccos(
                    xp.sin(lat_perturbation_center) * xp.sin(edge_lat)
                    + xp.cos(lat_perturbation_center)
                    * xp.cos(edge_lat)
                    * xp.cos(edge_lon - lon_perturbation_center)
                )
                ** 2
            ),
            u,
        )
    vn = u * primal_normal_x

    return vn
