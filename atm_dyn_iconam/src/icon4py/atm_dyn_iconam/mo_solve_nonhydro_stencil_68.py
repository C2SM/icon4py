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

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_68(
    mask_prog_halo_c: Field[[CellDim], bool],
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
        * ((exner_new / exner_now - 1.0) * cvd_o_rd + 1.0)
        / rho_new,
        theta_v_new,
    )
    return theta_v_new


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_68(
    mask_prog_halo_c: Field[[CellDim], bool],
    rho_now: Field[[CellDim, KDim], float],
    theta_v_now: Field[[CellDim, KDim], float],
    exner_new: Field[[CellDim, KDim], float],
    exner_now: Field[[CellDim, KDim], float],
    rho_new: Field[[CellDim, KDim], float],
    theta_v_new: Field[[CellDim, KDim], float],
    cvd_o_rd: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
