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
from gt4py.next.ffront.fbuiltins import Field

from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _mo_solve_nonhydro_stencil_11_upper(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    z_theta_v_pr_ic = (
        wgtfacq_c(Koff[-1]) * z_rth_pr(Koff[-1])
        + wgtfacq_c(Koff[-2]) * z_rth_pr(Koff[-2])
        + wgtfacq_c(Koff[-3]) * z_rth_pr(Koff[-3])
    )
    theta_v_ic = theta_ref_ic + z_theta_v_pr_ic
    return z_theta_v_pr_ic, theta_v_ic


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_11_upper(
    wgtfacq_c: Field[[CellDim, KDim], float],
    z_rth_pr: Field[[CellDim, KDim], float],
    theta_ref_ic: Field[[CellDim, KDim], float],
    z_theta_v_pr_ic: Field[[CellDim, KDim], float],
    theta_v_ic: Field[[CellDim, KDim], float],
):
    _mo_solve_nonhydro_stencil_11_upper(
        wgtfacq_c,
        z_rth_pr,
        theta_ref_ic,
        z_theta_v_pr_ic,
        out=(z_theta_v_pr_ic, theta_v_ic),
    )
