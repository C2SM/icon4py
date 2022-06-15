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
from functional.ffront.fbuiltins import Field, float

from icon4py.common.dimension import E2C, CellDim, EdgeDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_13(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_nabla2_e = kh_smag_e * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0]))
    return z_nabla2_e


@program
def mo_nh_diffusion_stencil_13(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_nabla2_e: Field[[EdgeDim, KDim], float],
):
    _mo_nh_diffusion_stencil_13(
        kh_smag_e,
        inv_dual_edge_length,
        theta_v,
        out=z_nabla2_e,
    )
