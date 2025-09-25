# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _apply_vertical_diffusion_to_vn(
    vn: fa.EdgeKField[wpfloat],
    ddqz_z_half_e: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    coeff: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    d2vndz2 = (
        (vn(Koff[-1]) - vn) / ddqz_z_half_e - (vn - vn(Koff[1])) / ddqz_z_half_e(Koff[1])
    ) / ddqz_z_full_e
    vn = vn + coeff * d2vndz2
    return vn


@field_operator
def _apply_vertical_diffusion_to_w(
    w: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    ddqz_z_full: fa.CellKField[vpfloat],
    coeff: wpfloat,
) -> fa.CellKField[wpfloat]:
    d2wdz2 = (
        (w(Koff[-1]) - w) / ddqz_z_full(Koff[-1]) - (w - w(Koff[1])) / ddqz_z_full
    ) / ddqz_z_half
    w = w + coeff * d2wdz2
    return w


@program(grid_type=GridType.UNSTRUCTURED)
def apply_vertical_diffusion_to_vn(
    vn: fa.EdgeKField[wpfloat],
    ddqz_z_half_e: fa.EdgeKField[vpfloat],
    ddqz_z_full_e: fa.EdgeKField[vpfloat],
    coeff: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_vertical_diffusion_to_vn(
        vn,
        ddqz_z_half_e,
        ddqz_z_full_e,
        coeff,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@program(grid_type=GridType.UNSTRUCTURED)
def apply_vertical_diffusion_to_w(
    w: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    ddqz_z_full: fa.CellKField[vpfloat],
    coeff: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _apply_vertical_diffusion_to_w(
        w,
        ddqz_z_half,
        ddqz_z_full,
        coeff,
        out=w,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
