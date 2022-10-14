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

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import max_over, maximum, min_over, minimum

from icon4py.common.dimension import C2E2C, C2E2CDim, CellDim, KDim


@field_operator
def _hflx_limiter_mo_stencil_03_min_max(
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
    beta_fct: float,
    r_beta_fct: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_max = beta_fct * maximum(
        max_over(z_tracer_max(C2E2C), axis=C2E2CDim), z_tracer_max
    )
    z_min = r_beta_fct * minimum(
        min_over(z_tracer_min(C2E2C), axis=C2E2CDim), z_tracer_min
    )
    return z_max, z_min


@program
def hflx_limiter_mo_stencil_03_min_max(
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
    beta_fct: float,
    r_beta_fct: float,
    z_max: Field[[CellDim, KDim], float],
    z_min: Field[[CellDim, KDim], float],
):
    _hflx_limiter_mo_stencil_03_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct, out=(z_max, z_min)
    )


@field_operator
def _hflx_limiter_mo_stencil_03(
    z_mflx_anti_in: Field[[CellDim, KDim], float],
    z_mflx_anti_out: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    z_max: Field[[CellDim, KDim], float],
    z_min: Field[[CellDim, KDim], float],
    dbl_eps: float,
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    r_p = (z_max - z_tracer_new_low) / (z_mflx_anti_in + dbl_eps)
    r_m = (z_tracer_new_low - z_min) / (z_mflx_anti_out + dbl_eps)
    return r_p, r_m


@program
def hflx_limiter_mo_stencil_03(
    z_tracer_max: Field[[CellDim, KDim], float],
    z_tracer_min: Field[[CellDim, KDim], float],
    beta_fct: float,
    r_beta_fct: float,
    z_max: Field[[CellDim, KDim], float],
    z_min: Field[[CellDim, KDim], float],
    z_mflx_anti_in: Field[[CellDim, KDim], float],
    z_mflx_anti_out: Field[[CellDim, KDim], float],
    z_tracer_new_low: Field[[CellDim, KDim], float],
    dbl_eps: float,
    r_p: Field[[CellDim, KDim], float],
    r_m: Field[[CellDim, KDim], float],
):

    _hflx_limiter_mo_stencil_03_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct, out=(z_min, z_max)
    )
    _hflx_limiter_mo_stencil_03(
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        z_max,
        z_min,
        dbl_eps,
        out=(r_p, r_m),
    )
