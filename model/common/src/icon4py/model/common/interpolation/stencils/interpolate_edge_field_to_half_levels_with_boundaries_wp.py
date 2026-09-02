# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.interpolation.stencils.interpolate_edge_field_to_half_levels_wp import (
    _interpolate_edge_field_to_half_levels_wp,
)
from icon4py.model.common.math.vertical_operations import (
    extrapolate_quadratically_to_surface_on_edges,
    extrapolate_quadratically_to_top_on_edges,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_edge_field_to_half_levels_with_boundaries_wp(
    interpolant: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
) -> fa.EdgeKField[wpfloat]:
    """
    Interpolate an edge field from full levels to half levels, including the
    quadratically extrapolated top and bottom boundary half levels.

    Boundary counterpart of ``_interpolate_edge_field_to_half_levels_wp``, which
    covers the interior half levels only.

    Args:
        interpolant: edge field on full levels (nlev levels)
        wgtfac_e: interpolation weight on half levels
        wgtfacq1_e: top extrapolation weights, one row per full level 0..2
        wgtfacq_e: bottom extrapolation weights, one row per full level
            nlev - 3..nlev - 1
        nlev: number of full levels

    Returns:
        edge field on half levels (nlev + 1 levels)
    """
    return concat_where(
        dims.KDim == 0,
        extrapolate_quadratically_to_top_on_edges(
            interpolant=interpolant,
            weights=wgtfacq1_e,
        ),
        concat_where(
            dims.KDim == nlev,
            extrapolate_quadratically_to_surface_on_edges(
                interpolant=interpolant,
                weights=wgtfacq_e,
            ),
            _interpolate_edge_field_to_half_levels_wp(wgtfac_e=wgtfac_e, interpolant=interpolant),
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_edge_field_to_half_levels_with_boundaries_wp(
    interpolant: fa.EdgeKField[wpfloat],
    wgtfac_e: fa.EdgeKField[wpfloat],
    wgtfacq1_e: fa.EdgeKField[wpfloat],
    wgtfacq_e: fa.EdgeKField[wpfloat],
    interpolation: fa.EdgeKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_edge_field_to_half_levels_with_boundaries_wp(
        interpolant=interpolant,
        wgtfac_e=wgtfac_e,
        wgtfacq1_e=wgtfacq1_e,
        wgtfacq_e=wgtfacq_e,
        nlev=nlev,
        out=interpolation,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
