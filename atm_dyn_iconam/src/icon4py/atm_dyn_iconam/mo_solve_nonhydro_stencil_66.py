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
from functional.ffront.fbuiltins import Field, exp, log, where

from icon4py.common.dimension import CellDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_66(
    bdy_halo_c: Field[[CellDim], bool],
    rho: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
    rd_o_p0ref: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    theta_v = where(bdy_halo_c, exner, theta_v)
    exner = where(bdy_halo_c, exp(rd_o_cvd * log(rd_o_p0ref * rho * exner)), exner)
    return theta_v, exner


@program
def mo_solve_nonhydro_stencil_66(
    bdy_halo_c: Field[[CellDim], bool],
    rho: Field[[CellDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    exner: Field[[CellDim, KDim], float],
    rd_o_cvd: float,
    rd_o_p0ref: float,
):

    _mo_solve_nonhydro_stencil_66(
        bdy_halo_c, rho, theta_v, exner, rd_o_cvd, rd_o_p0ref, out=(theta_v, exner)
    )
