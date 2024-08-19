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

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    _calculate_diagnostics_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    _temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CEDim, CellDim, KDim, KHalfDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostic_quantities_for_turbulence(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[[CEDim], wpfloat],
    geofac_div: gtx.Field[[CEDim], wpfloat],
    diff_multfac_smag: gtx.Field[[KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
) -> tuple[fa.CellKHalfField[vpfloat], fa.CellKHalfField[vpfloat]]:
    kh_c, div = _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic_vp, hdef_ic_vp = _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c)
    return div_ic_vp, hdef_ic_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_diagnostic_quantities_for_turbulence(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[[CEDim], wpfloat],
    geofac_div: gtx.Field[[CEDim], wpfloat],
    diff_multfac_smag: gtx.Field[[KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    div_ic: fa.CellKHalfField[vpfloat],
    hdef_ic: fa.CellKHalfField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _calculate_diagnostic_quantities_for_turbulence(
        kh_smag_ec,
        vn,
        e_bln_c_s,
        geofac_div,
        diff_multfac_smag,
        wgtfac_c,
        out=(div_ic, hdef_ic),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KHalfDim: (vertical_start, vertical_end + 1),
        },
    )
