# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_43."""
    ddt_w_adv_ntl1_wp, z_th_ddz_exner_c_wp, w_concorr_c_wp = astype(
        (ddt_w_adv_ntl1, ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, contravariant_correction_at_cells_on_half_levels), wpfloat
    )

    z_w_expl_wp = w_nnow + dtime * (ddt_w_adv_ntl1_wp - cpd * z_th_ddz_exner_c_wp)
    z_contr_w_fl_l_wp = rho_at_cells_on_half_levels * (-w_concorr_c_wp + exner_w_explicit_weight_parameter * w_nnow)
    return z_w_expl_wp, z_contr_w_fl_l_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    w_explicit_term: fa.CellKField[wpfloat],
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
        w_nnow,
        ddt_w_adv_ntl1,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        rho_at_cells_on_half_levels,
        contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter,
        dtime,
        cpd,
        out=(w_explicit_term, vertical_mass_flux_at_cells_on_half_levels),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
