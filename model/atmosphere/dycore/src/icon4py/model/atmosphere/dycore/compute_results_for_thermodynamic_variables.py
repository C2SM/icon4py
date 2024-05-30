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
from gt4py.next.ffront.fbuiltins import Field, astype, int32
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_results_for_thermodynamic_variables(
    z_rho_expl: fa.CKwpField,
    vwind_impl_wgt: fa.CwpField,
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    rho_ic: fa.CKwpField,
    w: fa.CKwpField,
    z_exner_expl: fa.CKwpField,
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    rho_now: fa.CKwpField,
    theta_v_now: fa.CKwpField,
    exner_now: fa.CKwpField,
    dtime: wpfloat,
    cvd_o_rd: wpfloat,
) -> tuple[
    fa.CKwpField,
    fa.CKwpField,
    fa.CKwpField,
]:
    """Formerly known as _mo_solve_nonhydro_stencil_55."""
    inv_ddqz_z_full_wp, exner_ref_mc_wp, z_alpha_wp, z_beta_wp = astype(
        (inv_ddqz_z_full, exner_ref_mc, z_alpha, z_beta), wpfloat
    )

    rho_new_wp = z_rho_expl - vwind_impl_wgt * dtime * inv_ddqz_z_full_wp * (
        rho_ic * w - rho_ic(Koff[1]) * w(Koff[1])
    )
    exner_new_wp = (
        z_exner_expl
        + exner_ref_mc_wp
        - z_beta_wp * (z_alpha_wp * w - z_alpha_wp(Koff[1]) * w(Koff[1]))
    )
    theta_v_new_wp = (
        rho_now
        * theta_v_now
        * ((exner_new_wp / exner_now - wpfloat("1.0")) * cvd_o_rd + wpfloat("1.0"))
        / rho_new_wp
    )
    return rho_new_wp, exner_new_wp, theta_v_new_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_results_for_thermodynamic_variables(
    z_rho_expl: fa.CKwpField,
    vwind_impl_wgt: fa.CwpField,
    inv_ddqz_z_full: Field[[CellDim, KDim], vpfloat],
    rho_ic: fa.CKwpField,
    w: fa.CKwpField,
    z_exner_expl: fa.CKwpField,
    exner_ref_mc: Field[[CellDim, KDim], vpfloat],
    z_alpha: Field[[CellDim, KDim], vpfloat],
    z_beta: Field[[CellDim, KDim], vpfloat],
    rho_now: fa.CKwpField,
    theta_v_now: fa.CKwpField,
    exner_now: fa.CKwpField,
    rho_new: fa.CKwpField,
    exner_new: fa.CKwpField,
    theta_v_new: fa.CKwpField,
    dtime: wpfloat,
    cvd_o_rd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_results_for_thermodynamic_variables(
        z_rho_expl,
        vwind_impl_wgt,
        inv_ddqz_z_full,
        rho_ic,
        w,
        z_exner_expl,
        exner_ref_mc,
        z_alpha,
        z_beta,
        rho_now,
        theta_v_now,
        exner_now,
        dtime,
        cvd_o_rd,
        out=(rho_new, exner_new, theta_v_new),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
