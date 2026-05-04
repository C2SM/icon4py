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
def _apply_2nd_order_divergence_damping(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    second_order_divdamp_scaling_coeff: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_26."""
    z_graddiv_vn_wp = astype(horizontal_gradient_of_normal_wind_divergence, wpfloat)

    vn_wp = vn + (second_order_divdamp_scaling_coeff * z_graddiv_vn_wp)
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_2nd_order_divergence_damping(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    second_order_divdamp_scaling_coeff: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_2nd_order_divergence_damping(
        horizontal_gradient_of_normal_wind_divergence,
        vn,
        second_order_divdamp_scaling_coeff,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
