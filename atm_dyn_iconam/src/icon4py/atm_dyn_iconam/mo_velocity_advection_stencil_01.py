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
from functional.ffront.fbuiltins import Field, float32, neighbor_sum

from icon4py.common.dimension import E2C2E, E2C2EDim, EdgeDim, KDim


@field_operator
def _mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float32],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float32],
) -> Field[[EdgeDim, KDim], float32]:
    vt = neighbor_sum(vn(E2C2E) * rbf_vec_coeff_e, axis=E2C2EDim)
    return vt


@program
def mo_velocity_advection_stencil_01(
    vn: Field[[EdgeDim, KDim], float32],
    rbf_vec_coeff_e: Field[[EdgeDim, E2C2EDim], float32],
    vt: Field[[EdgeDim, KDim], float32],
):
    _mo_velocity_advection_stencil_01(vn, rbf_vec_coeff_e, out=vt)
