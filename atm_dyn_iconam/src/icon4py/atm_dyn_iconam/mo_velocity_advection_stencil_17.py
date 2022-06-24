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

from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Field, neighbor_sum

from icon4py.common.dimension import C2E, C2EDim, CellDim, EdgeDim, KDim


@field_operator
def _mo_velocity_advection_stencil_17(
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    ddt_w_adv = ddt_w_adv + neighbor_sum(e_bln_c_s * z_v_grad_w(C2E), axis=C2EDim)
    return ddt_w_adv


@program
def mo_velocity_advection_stencil_17(
    e_bln_c_s: Field[[CellDim, C2EDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
):
    _mo_velocity_advection_stencil_17(e_bln_c_s, z_v_grad_w, ddt_w_adv, out=ddt_w_adv)
