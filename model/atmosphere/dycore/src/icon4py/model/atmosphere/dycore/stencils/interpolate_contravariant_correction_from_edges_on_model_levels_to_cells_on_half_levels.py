# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    contravariant_correction_at_cells_on_half_levels = concat_where(
        (nflatlev + 1 <= dims.KDim) & (dims.KDim < nlev),
        _compute_contravariant_correction_of_w(
            e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfac_c
        ),
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfacq_c
        ),
    )
    return contravariant_correction_at_cells_on_half_levels


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
        e_bln_c_s,
        contravariant_correction_at_edges_on_model_levels,
        wgtfac_c,
        wgtfacq_c,
        nlev,
        nflatlev,
        out=contravariant_correction_at_cells_on_half_levels,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
