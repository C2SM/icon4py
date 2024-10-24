# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.ffront.decorator import GridType, field_operator, program
from gt4py.next.ffront.fbuiltins import where

from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    vert_idx: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    w_concorr_c = where(
        nflatlev + 1 <= vert_idx < nlev,
        _compute_contravariant_correction_of_w(e_bln_c_s, z_w_concorr_me, wgtfac_c),
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, z_w_concorr_me, wgtfacq_c
        ),
    )
    return w_concorr_c


@program(grid_type=GridType.UNSTRUCTURED)
def fused_solve_nonhydro_stencil_39_40(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    vert_idx: fa.KField[gtx.int32],
    nlev: gtx.int32,
    nflatlev: gtx.int32,
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _fused_solve_nonhydro_stencil_39_40(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        wgtfacq_c,
        vert_idx,
        nlev,
        nflatlev,
        out=w_concorr_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
