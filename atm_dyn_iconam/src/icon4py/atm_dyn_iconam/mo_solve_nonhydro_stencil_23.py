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

from icon4py.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_23(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl2: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    dtime: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    cpd: float,
) -> Field[[EdgeDim, KDim], float]:
    vn_nnew = vn_nnow + dtime * (
        wgt_nnow_vel * ddt_vn_adv_ntl1
        + wgt_nnew_vel * ddt_vn_adv_ntl2
        + ddt_vn_phy
        - cpd * z_theta_v_e * z_gradh_exner
    )
    return vn_nnew


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_23(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_adv_ntl2: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    vn_nnew: Field[[EdgeDim, KDim], float],
    dtime: float,
    wgt_nnow_vel: float,
    wgt_nnew_vel: float,
    cpd: float,
):
    _mo_solve_nonhydro_stencil_23(
        vn_nnow,
        ddt_vn_adv_ntl1,
        ddt_vn_adv_ntl2,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        wgt_nnow_vel,
        wgt_nnew_vel,
        cpd,
        out=vn_nnew,
    )
