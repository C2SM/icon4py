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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_43."""
    ddt_w_adv_ntl1_wp, z_th_ddz_exner_c_wp, w_concorr_c_wp = astype(
        (ddt_w_adv_ntl1, z_th_ddz_exner_c, w_concorr_c), wpfloat
    )

    z_w_expl_wp = w_nnow + dtime * (ddt_w_adv_ntl1_wp - cpd * z_th_ddz_exner_c_wp)
    z_contr_w_fl_l_wp = rho_ic * (-w_concorr_c_wp + vwind_expl_wgt * w_nnow)
    return z_w_expl_wp, z_contr_w_fl_l_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    z_w_expl: fa.CellKField[wpfloat],
    w_nnow: fa.CellKField[wpfloat],
    ddt_w_adv_ntl1: fa.CellKField[vpfloat],
    z_th_ddz_exner_c: fa.CellKField[vpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
        w_nnow,
        ddt_w_adv_ntl1,
        z_th_ddz_exner_c,
        rho_ic,
        w_concorr_c,
        vwind_expl_wgt,
        dtime,
        cpd,
        out=(z_w_expl, z_contr_w_fl_l),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
