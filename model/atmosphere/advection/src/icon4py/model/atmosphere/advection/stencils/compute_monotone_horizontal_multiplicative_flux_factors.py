# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import max_over, maximum, min_over, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors_min_max(
    z_tracer_max: fa.CellKField[vpfloat],
    z_tracer_min: fa.CellKField[vpfloat],
    beta_fct: wpfloat,
    r_beta_fct: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    z_max = beta_fct * maximum(max_over(z_tracer_max(C2E2C), axis=dims.C2E2CDim), z_tracer_max)
    z_min = r_beta_fct * minimum(min_over(z_tracer_min(C2E2C), axis=dims.C2E2CDim), z_tracer_min)
    return z_max, z_min


@program
def compute_monotone_horizontal_multiplicative_flux_factors_min_max(
    z_tracer_max: fa.CellKField[vpfloat],
    z_tracer_min: fa.CellKField[vpfloat],
    beta_fct: wpfloat,
    r_beta_fct: wpfloat,
    z_max: fa.CellKField[vpfloat],
    z_min: fa.CellKField[vpfloat],
):
    _compute_monotone_horizontal_multiplicative_flux_factors_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct, out=(z_max, z_min)
    )


@field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors_a(
    z_mflx_anti_in: fa.CellKField[vpfloat],
    z_mflx_anti_out: fa.CellKField[vpfloat],
    z_tracer_new_low: fa.CellKField[wpfloat],
    z_max: fa.CellKField[vpfloat],
    z_min: fa.CellKField[vpfloat],
    dbl_eps: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    r_p = (z_max - z_tracer_new_low) / (z_mflx_anti_in + dbl_eps)
    r_m = (z_tracer_new_low - z_min) / (z_mflx_anti_out + dbl_eps)

    return r_p, r_m


@field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors(
    z_tracer_max: fa.CellKField[vpfloat],
    z_tracer_min: fa.CellKField[vpfloat],
    beta_fct: wpfloat,
    r_beta_fct: wpfloat,
    z_mflx_anti_in: fa.CellKField[vpfloat],
    z_mflx_anti_out: fa.CellKField[vpfloat],
    z_tracer_new_low: fa.CellKField[wpfloat],
    dbl_eps: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_max, z_min = _compute_monotone_horizontal_multiplicative_flux_factors_min_max(
        z_tracer_max, z_tracer_min, beta_fct, r_beta_fct
    )

    r_p, r_m = _compute_monotone_horizontal_multiplicative_flux_factors_a(
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        z_max,
        z_min,
        dbl_eps,
    )
    return r_p, r_m


@program
def compute_monotone_horizontal_multiplicative_flux_factors(
    z_tracer_max: fa.CellKField[vpfloat],
    z_tracer_min: fa.CellKField[vpfloat],
    beta_fct: wpfloat,
    r_beta_fct: wpfloat,
    z_mflx_anti_in: fa.CellKField[vpfloat],
    z_mflx_anti_out: fa.CellKField[vpfloat],
    z_tracer_new_low: fa.CellKField[wpfloat],
    dbl_eps: wpfloat,
    r_p: fa.CellKField[wpfloat],
    r_m: fa.CellKField[wpfloat],
):
    _compute_monotone_horizontal_multiplicative_flux_factors(
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
