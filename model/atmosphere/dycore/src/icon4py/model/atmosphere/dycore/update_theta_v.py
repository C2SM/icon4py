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
from gt4py.next.ffront.fbuiltins import Field, int32, where
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _update_theta_v(
    mask_prog_halo_c: fa.CboolField,
    rho_now: fa.CKwpField,
    theta_v_now: fa.CKwpField,
    exner_new: fa.CKwpField,
    exner_now: fa.CKwpField,
    rho_new: fa.CKwpField,
    theta_v_new: fa.CKwpField,
    cvd_o_rd: wpfloat,
) -> Field[[CellDim, KDim], wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_68."""
    theta_v_new_wp = where(
        mask_prog_halo_c,
        rho_now
        * theta_v_now
        * ((exner_new / exner_now - wpfloat("1.0")) * cvd_o_rd + wpfloat("1.0"))
        / rho_new,
        theta_v_new,
    )
    return theta_v_new_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def update_theta_v(
    mask_prog_halo_c: fa.CboolField,
    rho_now: fa.CKwpField,
    theta_v_now: fa.CKwpField,
    exner_new: fa.CKwpField,
    exner_now: fa.CKwpField,
    rho_new: fa.CKwpField,
    theta_v_new: fa.CKwpField,
    cvd_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _update_theta_v(
        mask_prog_halo_c,
        rho_now,
        theta_v_now,
        exner_new,
        exner_now,
        rho_new,
        theta_v_new,
        cvd_o_rd,
        out=theta_v_new,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
