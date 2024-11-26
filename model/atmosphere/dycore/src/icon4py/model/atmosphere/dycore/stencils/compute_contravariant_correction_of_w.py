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
from icon4py.model.common.dimension import C2CE, C2E, C2EDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_contravariant_correction_of_w(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_39."""
    z_w_concorr_me_offset_1 = z_w_concorr_me(Koff[-1])

    z_w_concorr_me_wp, z_w_concorr_me_offset_1_wp = astype(
        (z_w_concorr_me, z_w_concorr_me_offset_1), wpfloat
    )

    z_w_concorr_mc_m1_wp = neighbor_sum(
        e_bln_c_s(C2CE) * z_w_concorr_me_offset_1_wp(C2E), axis=C2EDim
    )
    z_w_concorr_mc_m0_wp = neighbor_sum(e_bln_c_s(C2CE) * z_w_concorr_me_wp(C2E), axis=C2EDim)

    z_w_concorr_mc_m1_vp, z_w_concorr_mc_m0_vp = astype(
        (z_w_concorr_mc_m1_wp, z_w_concorr_mc_m0_wp), vpfloat
    )
    w_concorr_c_vp = (
        wgtfac_c * z_w_concorr_mc_m0_vp + (vpfloat("1.0") - wgtfac_c) * z_w_concorr_mc_m1_vp
    )
    return w_concorr_c_vp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_contravariant_correction_of_w(
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CEDim], wpfloat],
    z_w_concorr_me: fa.EdgeKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_contravariant_correction_of_w(
        e_bln_c_s,
        z_w_concorr_me,
        wgtfac_c,
        out=w_concorr_c,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
