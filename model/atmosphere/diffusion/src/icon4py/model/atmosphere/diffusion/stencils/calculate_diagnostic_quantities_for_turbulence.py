# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    _calculate_diagnostics_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    _temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_diagnostic_quantities_for_turbulence(
    kh_smag: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKHalfField[vpfloat],
    w_old: fa.CellKHalfField[wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], wpfloat],
    compute_hgrad_w: bool,
    dwdx: fa.CellKHalfField[vpfloat],
    dwdy: fa.CellKHalfField[vpfloat],
) -> tuple[
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
    fa.CellKHalfField[vpfloat],
]:
    kh_c, div = _temporary_fields_for_turbulence_diagnostics(
        kh_smag, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic, hdef_ic = _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c)

    dwdx, dwdy = (
        _calculate_horizontal_gradients_for_turbulence(w_old, geofac_grg_x, geofac_grg_y)
        if compute_hgrad_w
        else (dwdx, dwdy)
    )

    return div_ic, hdef_ic, dwdx, dwdy
