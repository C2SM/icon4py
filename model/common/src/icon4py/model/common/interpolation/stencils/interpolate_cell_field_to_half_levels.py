# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _interpolate_cell_field_to_half_levels_vp(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    interpolant: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.vpfloat]:
    """
    Interpolate a CellDim variable of floating precision from full levels to half levels.
    The return variable also has floating precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_c: weight factor
        interpolant: CellDim variables at full levels
    Returns:
        CellDim variables at half levels
    """
    interpolation_to_half_levels_vp = wgtfac_c * interpolant + (
        vpfloat("1.0") - wgtfac_c
    ) * interpolant(Koff[-1])
    return interpolation_to_half_levels_vp


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _interpolate_cell_field_to_half_levels_wp(
    wgtfac_c: fa.CellKField[ta.wpfloat],
    interpolant: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
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


@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_surface_value_vp(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    interpolant: fa.CellKField[ta.vpfloat],
    surface_value: fa.CellKField[ta.vpfloat],
    surface_level: gtx.int32,
) -> fa.CellKField[ta.vpfloat]:
    """Interior linear interpolation to half levels for ``KDim < surface_level - 1``,
    caller-supplied ``surface_value`` at the surface."""
    return concat_where(
        dims.KDim < surface_level - 1,
        _interpolate_cell_field_to_half_levels_vp(wgtfac_c=wgtfac_c, interpolant=interpolant),
        surface_value,
    )


# TODO(havogt): Generics in GT4Py would allow to avoid
@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_surface_value_wp(
    wgtfac_c: fa.CellKField[ta.wpfloat],
    interpolant: fa.CellKField[ta.wpfloat],
    surface_value: fa.CellKField[ta.wpfloat],
    surface_level: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    """Interior linear interpolation to half levels for ``KDim < surface_level - 1``,
    caller-supplied ``surface_value`` at the surface."""
    return concat_where(
        dims.KDim < surface_level - 1,
        _interpolate_cell_field_to_half_levels_wp(wgtfac_c=wgtfac_c, interpolant=interpolant),
        surface_value,
    )
