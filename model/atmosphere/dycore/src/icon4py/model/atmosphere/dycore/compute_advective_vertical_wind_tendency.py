# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_advective_vertical_wind_tendency(
    z_w_con_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_16."""
    z_w_con_c_wp = astype(z_w_con_c, wpfloat)
    coeff1_dwdz_wp, coeff2_dwdz_wp = astype((coeff1_dwdz, coeff2_dwdz), wpfloat)

    ddt_w_adv_wp = -z_w_con_c_wp * (
        w(Koff[-1]) * coeff1_dwdz_wp
        - w(Koff[1]) * coeff2_dwdz_wp
        + w * astype(coeff2_dwdz - coeff1_dwdz, wpfloat)
    )
    return astype(ddt_w_adv_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_advective_vertical_wind_tendency(
    z_w_con_c: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    coeff1_dwdz: fa.CellKField[vpfloat],
    coeff2_dwdz: fa.CellKField[vpfloat],
    ddt_w_adv: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_advective_vertical_wind_tendency(
        z_w_con_c,
        w,
        coeff1_dwdz,
        coeff2_dwdz,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
