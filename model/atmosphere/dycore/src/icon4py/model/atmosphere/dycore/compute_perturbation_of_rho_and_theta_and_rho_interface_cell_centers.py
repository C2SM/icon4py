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

from icon4py.model.atmosphere.dycore.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
    wgtfac_c: fa.CKvpField,
    rho: fa.CKwpField,
    rho_ref_mc: fa.CKvpField,
    theta_v: fa.CKwpField,
    theta_ref_mc: fa.CKvpField,
) -> tuple[
    fa.CKwpField,
    fa.CKvpField,
    fa.CKvpField,
]:
    """Formerly known as _mo_solve_nonhydro_stencil_08."""
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    rho_ic = wgtfac_c_wp * rho + (wpfloat("1.0") - wgtfac_c_wp) * rho(Koff[-1])
    z_rth_pr_1, z_rth_pr_2 = _compute_perturbation_of_rho_and_theta(
        rho=rho, rho_ref_mc=rho_ref_mc, theta_v=theta_v, theta_ref_mc=theta_ref_mc
    )
    return rho_ic, z_rth_pr_1, z_rth_pr_2


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
    wgtfac_c: fa.CKvpField,
    rho: fa.CKwpField,
    rho_ref_mc: fa.CKvpField,
    theta_v: fa.CKwpField,
    theta_ref_mc: fa.CKvpField,
    rho_ic: fa.CKwpField,
    z_rth_pr_1: fa.CKvpField,
    z_rth_pr_2: fa.CKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
        wgtfac_c,
        rho,
        rho_ref_mc,
        theta_v,
        theta_ref_mc,
        out=(rho_ic, z_rth_pr_1, z_rth_pr_2),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
