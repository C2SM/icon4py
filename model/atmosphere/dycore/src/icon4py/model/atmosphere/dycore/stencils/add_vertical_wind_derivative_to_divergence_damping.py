# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, broadcast

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C, EdgeDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _add_vertical_wind_derivative_to_divergence_damping(
    horizontal_mask_for_3d_divdamp: fa.EdgeField[wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[vpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_17."""
    z_graddiv_vn_wp = astype(horizontal_gradient_of_normal_wind_divergence, wpfloat)

    scaling_factor_for_3d_divdamp = broadcast(scaling_factor_for_3d_divdamp, (EdgeDim, KDim))
    z_graddiv_vn_wp = z_graddiv_vn_wp + (
        horizontal_mask_for_3d_divdamp
        * scaling_factor_for_3d_divdamp
        * inv_dual_edge_length
        * astype(dwdz_at_cells_on_model_levels(E2C[1]) - dwdz_at_cells_on_model_levels(E2C[0]), wpfloat)
    )
    return astype(z_graddiv_vn_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_vertical_wind_derivative_to_divergence_damping(
    horizontal_mask_for_3d_divdamp: fa.EdgeField[wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[wpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[vpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_vertical_wind_derivative_to_divergence_damping(
        horizontal_mask_for_3d_divdamp,
        scaling_factor_for_3d_divdamp,
        inv_dual_edge_length,
        dwdz_at_cells_on_model_levels,
        horizontal_gradient_of_normal_wind_divergence,
        out=horizontal_gradient_of_normal_wind_divergence,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
