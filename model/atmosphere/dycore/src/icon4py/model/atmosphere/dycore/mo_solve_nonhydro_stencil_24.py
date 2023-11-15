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
from gt4py.next.ffront.fbuiltins import Field, int32

from icon4py.model.common.dimension import EdgeDim, KDim


@field_operator
def _mo_solve_nonhydro_stencil_24(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    dtime: float,
    cpd: float,
) -> Field[[EdgeDim, KDim], float]:
    vn_nnew = vn_nnow + dtime * (ddt_vn_apc_ntl1 - cpd * z_theta_v_e * z_gradh_exner + ddt_vn_phy)
    return vn_nnew


@program(grid_type=GridType.UNSTRUCTURED)
def mo_solve_nonhydro_stencil_24(
    vn_nnow: Field[[EdgeDim, KDim], float],
    ddt_vn_apc_ntl1: Field[[EdgeDim, KDim], float],
    ddt_vn_phy: Field[[EdgeDim, KDim], float],
    z_theta_v_e: Field[[EdgeDim, KDim], float],
    z_gradh_exner: Field[[EdgeDim, KDim], float],
    vn_nnew: Field[[EdgeDim, KDim], float],
    dtime: float,
    cpd: float,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _mo_solve_nonhydro_stencil_24(
        vn_nnow,
        ddt_vn_apc_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        cpd,
        out=vn_nnew,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
