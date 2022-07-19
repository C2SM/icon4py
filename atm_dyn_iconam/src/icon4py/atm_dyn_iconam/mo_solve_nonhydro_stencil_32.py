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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_32_mass_fl_e(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_vn_avg: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    mass_fl_e = z_rho_e * z_vn_avg * ddqz_z_full_e
    return mass_fl_e


@field_operator
def _mo_solve_nonhydro_stencil_32_z_theta_v_fl_e(
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_theta_v_fl_e = mass_fl_e * z_theta_v_e
    return z_theta_v_fl_e


@program
def mo_solve_nonhydro_stencil_32(
    z_rho_e: Field[[EdgeDim, KDim], float],
    z_vn_avg: Field[[EdgeDim, KDim], float],
    ddqz_z_full_e: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    mass_fl_e: Field[[EdgeDim, KDim], float],
    z_theta_v_fl_e: Field[[EdgeDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_32_mass_fl_e(
        z_rho_e, z_vn_avg, ddqz_z_full_e, out=mass_fl_e
    )
    _mo_solve_nonhydro_stencil_32_z_theta_v_fl_e(
        mass_fl_e, z_theta_v_e, out=z_theta_v_fl_e
    )
