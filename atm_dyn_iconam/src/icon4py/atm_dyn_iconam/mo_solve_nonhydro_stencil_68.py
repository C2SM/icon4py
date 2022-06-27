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

# TODO: conditionals not yet implemented.

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, where

from icon4py.common.dimension import CellDim, KDim


# TODO: check if the return is ok for the else as well as how the if statement is evaluating to True
@field_operator
def _mo_solve_nonhydro_stencil_68(
    mask_prog_halo_c: Field[[CellDim], float],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    cvd_o_rd: float,
) -> Field[[CellDim, KDim], float]:
    theta_v_new = where(
        mask_prog_halo_c,
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - 1) * cvd_o_rd + 1.0)
        / rho_new,
        theta_v_new,
    )
    return theta_v_new


@program
def mo_solve_nonhydro_stencil_68(
    mask_prog_halo_c: Field[[CellDim], float],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    cvd_o_rd: float,
):
    _mo_solve_nonhydro_stencil_68(
        mask_prog_halo_c,
        rho_now,
        theta_v_now,
        exner_new,
        exner_now,
        rho_new,
        theta_v_new,
        cvd_o_rd,
        out=theta_v_new,
    )
