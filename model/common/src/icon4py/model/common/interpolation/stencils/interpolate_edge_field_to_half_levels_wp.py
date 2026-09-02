# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_edge_field_to_half_levels_wp(
    wgtfac_e: fa.EdgeKField[ta.wpfloat],
    interpolant: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    """
    Interpolate a EdgeDim variable of working precision from full levels to half levels.
    The return variable also has working precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_e: weight factor
        interpolant: EdgeDim variables at full levels
    Returns:
        EdgeDim variables at half levels
    """
    interpolation_to_half_levels_wp = wgtfac_e * interpolant + (
        wpfloat("1.0") - wgtfac_e
    ) * interpolant(KDim - 1)
    return interpolation_to_half_levels_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_edge_field_to_half_levels_wp(
    wgtfac_e: fa.EdgeKField[ta.wpfloat],
    interpolant: fa.EdgeKField[ta.wpfloat],
    interpolation_to_half_levels_wp: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_edge_field_to_half_levels_wp(
        wgtfac_e=wgtfac_e,
        interpolant=interpolant,
        out=interpolation_to_half_levels_wp,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
