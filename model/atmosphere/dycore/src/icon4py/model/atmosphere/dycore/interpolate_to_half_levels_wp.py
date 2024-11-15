# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _interpolate_to_half_levels_wp(
    wgtfac_c: fa.CellKField[wpfloat],
    interpolant: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Interpolate a CellDim variable of working precision from full levels to half levels.
    The return variable also has working precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_c: weight factor
        interpolant: CellDim variables at full levels
    Returns:
        CellDim variables at half levels
    """
    interpolation_to_half_levels_wp = wgtfac_c * interpolant + (
        wpfloat("1.0") - wgtfac_c
    ) * interpolant(Koff[-1])
    return interpolation_to_half_levels_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_to_half_levels_wp(
    wgtfac_c: fa.CellKField[wpfloat],
    interpolant: fa.CellKField[wpfloat],
    interpolation_to_half_levels_wp: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_to_half_levels_wp(
        wgtfac_c,
        interpolant,
        out=interpolation_to_half_levels_wp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
