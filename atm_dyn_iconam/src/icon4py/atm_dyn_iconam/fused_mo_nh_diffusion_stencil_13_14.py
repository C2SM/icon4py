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

from icon4py.common.dimension import (
    E2C,
    E2CDim,
    C2E2C,
    C2E2CDim,
    C2CE,
    C2E,
    C2EDim,
    CEDim,
    CellDim,
    EdgeDim,
    KDim,
)

@field_operator
def _mo_nh_diffusion_stencil_13(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
) -> Field[[EdgeDim, KDim], float]:
    z_nabla2_e = kh_smag_e * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0]))
    return z_nabla2_e

@field_operator
def _mo_nh_diffusion_stencil_14(
    z_nabla2_e: Field[[EdgeDim, KDim], float],
    geofac_div: Field[[CEDim], float],
) -> Field[[CellDim, KDim], float]:
    z_temp = neighbor_sum(z_nabla2_e(C2E) * geofac_div(C2CE), axis=C2EDim)
    return z_temp

@field_operator
def _fused_mo_nh_diffusion_stencil_13_14(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
) -> Field[[CellDim, KDim], float]:
    z_nabla2_e = _mo_nh_diffusion_stencil_13(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _mo_nh_diffusion_stencil_14(z_nabla2_e, geofac_div)
    return z_temp

@program
def fused_mo_nh_diffusion_stencil_13_14(
    kh_smag_e: Field[[EdgeDim, KDim], float],
    inv_dual_edge_length: Field[[EdgeDim], float],
    theta_v: Field[[CellDim, KDim], float],
    geofac_div: Field[[CEDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    _fused_mo_nh_diffusion_stencil_13_14(kh_smag_e, inv_dual_edge_length, theta_v, geofac_div, out=z_temp)
