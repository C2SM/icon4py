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

from icon4py.common.dimension import C2E, C2CE, C2EDim, CellDim, EdgeDim, CEDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_14(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CEDim], float],
) -> Field[[CellDim, KDim], float]:
    z_temp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)
    return z_temp


@program
def mo_nh_diffusion_stencil_14(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_14(z_nabla2_e, geofac_div, out=z_temp)
