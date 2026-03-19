# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.dycore_utils import (
    _calculate_fourth_order_divdamp_scaling_coeff,
    _calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.constants import WP_EPS


@gtx.field_operator
def _apply_weighted_2nd_and_4th_order_divergence_damping(
    interpolated_fourth_order_divdamp_factor: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    divdamp_order: gtx.int32,
    mean_cell_area: wpfloat,
    second_order_divdamp_factor: wpfloat,
    max_nudging_coefficient: wpfloat,
    wp_eps: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_27."""
    scal_divdamp = _calculate_fourth_order_divdamp_scaling_coeff(
        interpolated_fourth_order_divdamp_factor,
        divdamp_order,
        mean_cell_area,
        second_order_divdamp_factor,
    )
    bdy_divdamp = _calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary(
        scal_divdamp, max_nudging_coefficient, wp_eps
    )
    z_graddiv2_vn_wp = astype(z_graddiv2_vn, wpfloat)
    vn_wp = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn_wp
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_weighted_2nd_and_4th_order_divergence_damping(
    interpolated_fourth_order_divdamp_factor: fa.KField[wpfloat],
    nudgecoeff_e: fa.EdgeField[wpfloat],
    z_graddiv2_vn: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    divdamp_order: gtx.int32,
    mean_cell_area: wpfloat,
    second_order_divdamp_factor: wpfloat,
    max_nudging_coefficient: wpfloat,
    wp_eps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_weighted_2nd_and_4th_order_divergence_damping(
        interpolated_fourth_order_divdamp_factor,
        nudgecoeff_e,
        z_graddiv2_vn,
        vn,
        divdamp_order,
        mean_cell_area,
        second_order_divdamp_factor,
        max_nudging_coefficient,
        WP_EPS,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
