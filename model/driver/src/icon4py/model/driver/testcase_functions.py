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

from icon4py.model.common.constants import CVD_O_RD, P0REF, RD
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
        rho[:, k] = exner[:, k] ** CVD_O_RD * P0REF / (RD * theta_v[:, k])

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
        rho[:, k] = exner[:, k] ** CVD_O_RD * P0REF / (RD * theta_v[:, k])

    return rho, exner
