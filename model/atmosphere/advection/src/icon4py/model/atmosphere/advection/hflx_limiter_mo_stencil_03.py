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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import max_over, maximum, min_over, minimum

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, C2E2CDim


@field_operator
def _hflx_limiter_mo_stencil_03_min_max(
    z_tracer_max: fa.CKfloatField,
    z_tracer_min: fa.CKfloatField,
    beta_fct: float,
    r_beta_fct: float,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    z_max = beta_fct * maximum(max_over(z_tracer_max(C2E2C), axis=C2E2CDim), z_tracer_max)
    z_min = r_beta_fct * minimum(min_over(z_tracer_min(C2E2C), axis=C2E2CDim), z_tracer_min)
    return z_max, z_min


@program
def hflx_limiter_mo_stencil_03_min_max(
    z_tracer_max: fa.CKfloatField,
    z_tracer_min: fa.CKfloatField,
    beta_fct: float,
    r_beta_fct: float,
    z_max: fa.CKfloatField,
    z_min: fa.CKfloatField,
):
    _hflx_limiter_mo_stencil_03_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct, out=(z_max, z_min)
    )


@field_operator
def _hflx_limiter_mo_stencil_03a(
    z_mflx_anti_in: fa.CKfloatField,
    z_mflx_anti_out: fa.CKfloatField,
    z_tracer_new_low: fa.CKfloatField,
    z_max: fa.CKfloatField,
    z_min: fa.CKfloatField,
    dbl_eps: float,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    r_p = (z_max - z_tracer_new_low) / (z_mflx_anti_in + dbl_eps)
    r_m = (z_tracer_new_low - z_min) / (z_mflx_anti_out + dbl_eps)

    return r_p, r_m


@field_operator
def _hflx_limiter_mo_stencil_03(
    z_tracer_max: fa.CKfloatField,
    z_tracer_min: fa.CKfloatField,
    beta_fct: float,
    r_beta_fct: float,
    z_mflx_anti_in: fa.CKfloatField,
    z_mflx_anti_out: fa.CKfloatField,
    z_tracer_new_low: fa.CKfloatField,
    dbl_eps: float,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    z_max, z_min = _hflx_limiter_mo_stencil_03_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct
    )

    r_p, r_m = _hflx_limiter_mo_stencil_03a(
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        z_max,
        z_min,
        dbl_eps,
    )
    return r_p, r_m


@program
def hflx_limiter_mo_stencil_03(
    z_tracer_max: fa.CKfloatField,
    z_tracer_min: fa.CKfloatField,
    beta_fct: float,
    r_beta_fct: float,
    z_mflx_anti_in: fa.CKfloatField,
    z_mflx_anti_out: fa.CKfloatField,
    z_tracer_new_low: fa.CKfloatField,
    dbl_eps: float,
    r_p: fa.CKfloatField,
    r_m: fa.CKfloatField,
):
    _hflx_limiter_mo_stencil_03(
        z_tracer_max,
        z_tracer_min,
        beta_fct,
        r_beta_fct,
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        dbl_eps,
        out=(r_p, r_m),
    )
