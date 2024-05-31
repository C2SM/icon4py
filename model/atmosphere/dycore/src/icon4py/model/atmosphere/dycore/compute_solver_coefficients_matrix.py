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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_solver_coefficients_matrix(
    exner_nnow: fa.CKwpField,
    rho_nnow: fa.CKwpField,
    theta_v_nnow: fa.CKwpField,
    inv_ddqz_z_full: fa.CKvpField,
    vwind_impl_wgt: fa.CwpField,
    theta_v_ic: fa.CKwpField,
    rho_ic: fa.CKwpField,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
) -> tuple[fa.CKvpField, fa.CKvpField]:
    """Formerly known as _mo_solve_nonhydro_stencil_44."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_beta_wp = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full_wp
    z_alpha_wp = vwind_impl_wgt * theta_v_ic * rho_ic
    return astype((z_beta_wp, z_alpha_wp), vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_solver_coefficients_matrix(
    z_beta: fa.CKvpField,
    exner_nnow: fa.CKwpField,
    rho_nnow: fa.CKwpField,
    theta_v_nnow: fa.CKwpField,
    inv_ddqz_z_full: fa.CKvpField,
    z_alpha: fa.CKvpField,
    vwind_impl_wgt: fa.CwpField,
    theta_v_ic: fa.CKwpField,
    rho_ic: fa.CKwpField,
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_solver_coefficients_matrix(
        exner_nnow,
        rho_nnow,
        theta_v_nnow,
        inv_ddqz_z_full,
        vwind_impl_wgt,
        theta_v_ic,
        rho_ic,
        dtime,
        rd,
        cvd,
        out=(z_beta, z_alpha),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
