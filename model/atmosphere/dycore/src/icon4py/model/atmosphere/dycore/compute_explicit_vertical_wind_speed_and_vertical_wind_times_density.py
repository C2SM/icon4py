# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import astype, int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    w_nnow: fa.CKwpField,
    ddt_w_adv_ntl1: fa.CKvpField,
    z_th_ddz_exner_c: fa.CKvpField,
    rho_ic: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
    vwind_expl_wgt: fa.CwpField,
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CKwpField, fa.CKwpField]:
    """Formerly known as _mo_solve_nonhydro_stencil_43."""
    ddt_w_adv_ntl1_wp, z_th_ddz_exner_c_wp, w_concorr_c_wp = astype(
        (ddt_w_adv_ntl1, z_th_ddz_exner_c, w_concorr_c), wpfloat
    )

    z_w_expl_wp = w_nnow + dtime * (ddt_w_adv_ntl1_wp - cpd * z_th_ddz_exner_c_wp)
    z_contr_w_fl_l_wp = rho_ic * (-w_concorr_c_wp + vwind_expl_wgt * w_nnow)
    return z_w_expl_wp, z_contr_w_fl_l_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_explicit_vertical_wind_speed_and_vertical_wind_times_density(
    z_w_expl: fa.CKwpField,
    w_nnow: fa.CKwpField,
    ddt_w_adv_ntl1: fa.CKvpField,
    z_th_ddz_exner_c: fa.CKvpField,
    z_contr_w_fl_l: fa.CKwpField,
    rho_ic: fa.CKwpField,
    w_concorr_c: fa.CKvpField,
    vwind_expl_wgt: fa.CwpField,
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
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
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
