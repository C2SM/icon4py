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
from functional.ffront.fbuiltins import Field, float32

from icon4py.common.dimension import EdgeDim, KDim


# TODO: globals not yet implemented.
fac_bdy_diff = float32(5.0)


@field_operator
def _mo_nh_diffusion_stencil_06(
    z_nabla2_e: Field[[EdgeDim, KDim], float32],
    area_edge: Field[[EdgeDim], float32],
    vn: Field[[EdgeDim, KDim], float32],
) -> Field[[EdgeDim, KDim], float32]:
    vn = vn + (z_nabla2_e * area_edge * float32(5.0))
    return vn


@program
def mo_nh_diffusion_stencil_06(
    z_nabla2_e: Field[[EdgeDim, KDim], float32],
    area_edge: Field[[EdgeDim], float32],
    vn: Field[[EdgeDim, KDim], float32],
):
    _mo_nh_diffusion_stencil_06(z_nabla2_e, area_edge, vn, out=vn)
