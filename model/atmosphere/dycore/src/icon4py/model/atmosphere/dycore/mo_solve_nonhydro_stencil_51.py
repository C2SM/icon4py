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
from gt4py.next.ffront.fbuiltins import int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.settings import backend


@field_operator
def _mo_solve_nonhydro_stencil_51(
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    ddqz_z_half: fa.CKfloatField,
    z_beta: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    z_w_expl: fa.CKfloatField,
    z_exner_expl: fa.CKfloatField,
    dtime: float,
    cpd: float,
) -> tuple[fa.CKfloatField, fa.CKfloatField]:
    z_gamma = dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half
    z_c = -z_gamma * z_beta * z_alpha(Koff[1])
    z_b = 1.0 + z_gamma * z_alpha * (z_beta(Koff[-1]) + z_beta)
    z_q = -z_c / z_b
    w_nnew = z_w_expl - z_gamma * (z_exner_expl(Koff[-1]) - z_exner_expl)
    w_nnew = w_nnew / z_b

    return z_q, w_nnew


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def mo_solve_nonhydro_stencil_51(
    z_q: fa.CKfloatField,
    w_nnew: fa.CKfloatField,
    vwind_impl_wgt: fa.CfloatField,
    theta_v_ic: fa.CKfloatField,
    ddqz_z_half: fa.CKfloatField,
    z_beta: fa.CKfloatField,
    z_alpha: fa.CKfloatField,
    z_w_expl: fa.CKfloatField,
    z_exner_expl: fa.CKfloatField,
    dtime: float,
    cpd: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_51(
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_beta,
        z_alpha,
        z_w_expl,
        z_exner_expl,
        dtime,
        cpd,
        out=(z_q, w_nnew),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
