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
from icon4py.model.common.math.vertical_operations import (
    extrapolate_quadratically_to_surface_on_cells,
    extrapolate_quadratically_to_top_on_cells,
    with_boundaries_on_half_levels_on_cells,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _interpolate_cell_field_to_half_levels_vp(
    interpolant: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
) -> fa.CellKHalfField[ta.vpfloat]:
    """
    Interpolate a CellDim variable of floating precision from full levels to half levels.
    The return variable also has floating precision.
        var_half_k = wgtfac_c_k * var_full_k + (1 - wgtfac_c_k) * var_full_k-1
    (half level k lies above full level k, so ``var_full_k`` is the level below it)

    Args:
        interpolant: CellDim variables at full levels
        wgtfac_c: weight factor
    Returns:
        CellDim variables at half levels
    """
    interpolation_to_half_levels_vp = wgtfac_c * interpolant(dims.KHalfDim + 0.5) + (
        vpfloat("1.0") - wgtfac_c
    ) * interpolant(dims.KHalfDim - 0.5)
    return interpolation_to_half_levels_vp


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def _interpolate_cell_field_to_half_levels_wp(
    interpolant: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.wpfloat],
) -> fa.CellKHalfField[ta.wpfloat]:
    """
    Interpolate a CellDim variable of working precision from full levels to half levels.
    The return variable also has working precision.
        var_half_k = wgtfac_c_k * var_full_k + (1 - wgtfac_c_k) * var_full_k-1
    (half level k lies above full level k, so ``var_full_k`` is the level below it)

    Args:
        interpolant: CellDim variables at full levels
        wgtfac_c: weight factor
    Returns:
        CellDim variables at half levels
    """
    interpolation_to_half_levels_wp = wgtfac_c * interpolant(dims.KHalfDim + 0.5) + (
        wpfloat("1.0") - wgtfac_c
    ) * interpolant(dims.KHalfDim - 0.5)
    return interpolation_to_half_levels_wp


@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_surface_value_vp(
    interpolant: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKHalfField[ta.vpfloat],
    surface_value: fa.CellKHalfField[ta.vpfloat],
    surface_level: gtx.int32,
) -> fa.CellKHalfField[ta.vpfloat]:
    """Interior linear interpolation to half levels for ``KDim < surface_level - 1``,
    caller-supplied ``surface_value`` at the surface."""
    return concat_where(
        dims.KHalfDim < surface_level - 1,
        _interpolate_cell_field_to_half_levels_vp(interpolant=interpolant, wgtfac_c=wgtfac_c),
        surface_value,
    )


# TODO(havogt): Generics in GT4Py would allow to avoid
@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_surface_value_wp(
    interpolant: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKHalfField[ta.wpfloat],
    surface_value: fa.CellKHalfField[ta.wpfloat],
    surface_level: gtx.int32,
) -> fa.CellKHalfField[ta.wpfloat]:
    """Interior linear interpolation to half levels for ``KDim < surface_level - 1``,
    caller-supplied ``surface_value`` at the surface."""
    return concat_where(
        dims.KHalfDim < surface_level - 1,
        _interpolate_cell_field_to_half_levels_wp(interpolant=interpolant, wgtfac_c=wgtfac_c),
        surface_value,
    )


@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_boundaries(
    interpolant: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKHalfField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKHalfField[wpfloat]:
    """
    Interpolate a cell field from full levels to half levels, with the top and bottom
    half levels extrapolated quadratically.

    Args:
        interpolant: cell field on full levels
        wgtfac_c: interpolation weight on half levels
        wgtfacq1_c: top extrapolation weights, one row per full level 0..2
        wgtfacq_c: bottom extrapolation weights, one row per full level nlev - 3..nlev - 1
        nlev: number of full levels

    Returns:
        cell field on half levels
    """
    return with_boundaries_on_half_levels_on_cells(
        top=extrapolate_quadratically_to_top_on_cells(interpolant=interpolant, weights=wgtfacq1_c),
        interior=_interpolate_cell_field_to_half_levels_wp(
            interpolant=interpolant, wgtfac_c=wgtfac_c
        ),
        bottom=extrapolate_quadratically_to_surface_on_cells(
            interpolant=interpolant, weights=wgtfacq_c
        ),
        nlev=nlev,
    )
