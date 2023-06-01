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

from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, neighbor_sum
from icon4py.model.common.dimension import E2C2E, E2C2EDim, EdgeDim, KDim


@field_operator
def _mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
) -> Field[[EdgeDim, KDim], float]:
    vt = neighbor_sum(vn(E2C2E) * rbf_vec_coeff_e, axis=E2C2EDim)
    return vt


@program
def mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float],
    vt: Field[[EdgeDim, KDim], float],
):
    _mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e, out=vt)
