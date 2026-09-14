# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.math.vertical_operations import (
    extrapolate_quadratically_to_surface_on_edges,
    extrapolate_quadratically_to_top_on_edges,
    with_boundaries_on_half_levels_on_edges,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_edge_field_to_half_levels(
    wgtfac_e: fa.EdgeKHalfField[wpfloat],
    interpolant: fa.EdgeKField[wpfloat],
) -> fa.EdgeKHalfField[wpfloat]:
    """
    Interpolate an edge field from full levels to half levels.
        var_half_k = wgtfac_e_k * var_full_k + (1 - wgtfac_e_k) * var_full_k-1
    (half level k lies above full level k, so ``var_full_k`` is the level below it)

    Args:
        wgtfac_e: weight factor
        interpolant: edge field on full levels
    Returns:
        edge field on half levels
    """
    return wgtfac_e * interpolant(dims.KHalfDim + 0.5) + (wpfloat("1.0") - wgtfac_e) * interpolant(
        dims.KHalfDim - 0.5
    )


@gtx.field_operator
def _interpolate_edge_field_to_half_levels_with_boundaries(
    interpolant: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKHalfField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
) -> fa.EdgeKHalfField[wpfloat]:
    """
    Interpolate an edge field from full levels to half levels, with the top and bottom
    half levels extrapolated quadratically.

    Args:
        interpolant: edge field on full levels
        wgtfac_e: interpolation weight on half levels
        wgtfacq1_e: top extrapolation weights, one row per full level 0..2
        wgtfacq_e: bottom extrapolation weights, one row per full level nlev - 3..nlev - 1
        nlev: number of full levels

    Returns:
        edge field on half levels
    """
    return with_boundaries_on_half_levels_on_edges(
        top=extrapolate_quadratically_to_top_on_edges(interpolant=interpolant, weights=wgtfacq1_e),
        interior=_interpolate_edge_field_to_half_levels(wgtfac_e=wgtfac_e, interpolant=interpolant),
        bottom=extrapolate_quadratically_to_surface_on_edges(
            interpolant=interpolant, weights=wgtfacq_e
        ),
        nlev=nlev,
    )
