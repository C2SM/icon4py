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
from gt4py.next.ffront.fbuiltins import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CE, C2E, C2EDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    kh_smag_ec_wp, diff_multfac_smag_wp = astype((kh_smag_ec, diff_multfac_smag), wpfloat)

    kh_c_wp = neighbor_sum(kh_smag_ec_wp(C2E) * e_bln_c_s(C2CE), axis=C2EDim) / diff_multfac_smag_wp
    div_wp = neighbor_sum(vn(C2E) * geofac_div(C2CE), axis=C2EDim)
    return astype((kh_c_wp, div_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def temporary_fields_for_turbulence_diagnostics(
    kh_smag_ec: fa.EdgeKField[vpfloat],
    vn: fa.EdgeKField[wpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    diff_multfac_smag: gtx.Field[gtx.Dims[dims.KDim], vpfloat],
    kh_c: fa.CellKField[vpfloat],
    div: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _temporary_fields_for_turbulence_diagnostics(
        kh_smag_ec,
        vn,
        e_bln_c_s,
        geofac_div,
        diff_multfac_smag,
        out=(kh_c, div),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
