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
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum

from icon4py.model.common.dimension import C2CE, C2E, C2EDim, CEDim, CellDim, EdgeDim, KDim


@field_operator
def _mo_velocity_advection_stencil_17(
    e_bln_c_s: Field[[CEDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    ddt_w_adv = ddt_w_adv + neighbor_sum(z_v_grad_w(C2E) * e_bln_c_s(C2CE), axis=C2EDim)
    return ddt_w_adv


@program(grid_type=GridType.UNSTRUCTURED)
def mo_velocity_advection_stencil_17(
    e_bln_c_s: Field[[CEDim], float],
    z_v_grad_w: Field[[EdgeDim, KDim], float],
    ddt_w_adv: Field[[CellDim, KDim], float],
):
    _mo_velocity_advection_stencil_17(e_bln_c_s, z_v_grad_w, ddt_w_adv, out=ddt_w_adv)
