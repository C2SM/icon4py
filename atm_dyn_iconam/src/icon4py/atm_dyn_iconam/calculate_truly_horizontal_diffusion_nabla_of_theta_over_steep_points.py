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

from functional.ffront.fbuiltins import neighbor_sum
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, as_offset, where

from icon4py.common.dimension import C2E2C, C2E2CDim, CellDim, KDim, Koff


@field_operator
def _calculate_truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: Field[[CellDim, KDim], bool],
    zd_vertidx: Field[[CellDim, C2E2CDim, KDim], float],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CellDim, C2E2CDim], float],
    vcoef: Field[[CellDim, C2E2CDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:
    theta_v_offset = theta_v(C2E2C)(as_offset(Koff, zd_vertidx))
    theta_v_offset_1 = theta_v(C2E2C)(as_offset(Koff, (zd_vertidx + 1.0)))
    expr_1 = vcoef * theta_v_offset + (1.0 - vcoef) * theta_v_offset_1
    sum_1 = neighbor_sum(geofac_n2s_nbh * expr_1, axis=C2E2CDim)
    z_temp_expr = z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum_1)
    return where(mask, z_temp_expr, z_temp)


@program
def calculate_truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: Field[[CellDim, KDim], bool],
    zd_vertidx: Field[[CellDim, C2E2CDim, KDim], float],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CellDim, C2E2CDim], float],
    vcoef: Field[[CellDim, C2E2CDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    _calculate_truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
        mask,
        zd_vertidx,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v,
        z_temp,
        out=z_temp,
    )
