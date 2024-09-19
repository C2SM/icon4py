# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    _calculate_diagnostics_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.temporary_fields_for_turbulence_diagnostics import (
    _temporary_fields_for_turbulence_diagnostics,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _calculate_diagnostic_quantities_for_turbulence(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    geofac_div: Field[[dims.CEDim], wpfloat],
    diff_multfac_smag: Field[[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    kh_c, div = _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec, vn, e_bln_c_s, geofac_div, diff_multfac_smag
    )
    div_ic_vp, hdef_ic_vp = _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c)
    return div_ic_vp, hdef_ic_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def calculate_diagnostic_quantities_for_turbulence(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: Field[[dims.CEDim], wpfloat],
    geofac_div: Field[[dims.CEDim], wpfloat],
    diff_multfac_smag: Field[[dims.KDim], vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    div_ic: fa.CellKField[vpfloat],
    hdef_ic: fa.CellKField[vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
