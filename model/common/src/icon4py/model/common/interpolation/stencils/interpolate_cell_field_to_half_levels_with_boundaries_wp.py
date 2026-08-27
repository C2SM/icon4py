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
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.math.vertical_operations import (
    extrapolate_quadratically_to_surface_on_cells,
    extrapolate_quadratically_to_top_on_cells,
)
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _interpolate_cell_field_to_half_levels_with_boundaries_wp(
    interpolant: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Interpolate a cell field from full levels to half levels, including the
    quadratically extrapolated top and bottom boundary half levels.

    Boundary counterpart of ``_interpolate_cell_field_to_half_levels_wp``, which
    covers the interior half levels only.

    Args:
        interpolant: cell field on full levels (nlev levels)
        wgtfac_c: interpolation weight on half levels
        wgtfacq1_c: top extrapolation weights, one row per full level 0..2
        wgtfacq_c: bottom extrapolation weights, one row per full level
            nlev - 3..nlev - 1
        nlev: number of full levels

    Returns:
        cell field on half levels (nlev + 1 levels)
    """
    return concat_where(
        dims.KDim == 0,
        extrapolate_quadratically_to_top_on_cells(
            interpolant=interpolant,
            weights=wgtfacq1_c,
        ),
        concat_where(
            dims.KDim == nlev,
            extrapolate_quadratically_to_surface_on_cells(
                interpolant=interpolant,
                weights=wgtfacq_c,
            ),
            _interpolate_cell_field_to_half_levels_wp(wgtfac_c=wgtfac_c, interpolant=interpolant),
        ),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_cell_field_to_half_levels_with_boundaries_wp(
    interpolant: fa.CellKField[wpfloat],
    wgtfac_c: fa.CellKField[wpfloat],
    wgtfacq1_c: fa.CellKField[wpfloat],
    wgtfacq_c: fa.CellKField[wpfloat],
    interpolation: fa.CellKField[wpfloat],
    nlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_cell_field_to_half_levels_with_boundaries_wp(
        interpolant=interpolant,
        wgtfac_c=wgtfac_c,
        wgtfacq1_c=wgtfacq1_c,
        wgtfacq_c=wgtfacq_c,
        nlev=nlev,
        out=interpolation,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
