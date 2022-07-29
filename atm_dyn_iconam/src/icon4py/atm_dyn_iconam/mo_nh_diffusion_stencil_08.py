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

from icon4py.common.dimension import C2E2CO, C2E2CODim, CellDim, KDim


@field_operator
def _mo_nh_diffusion_stencil_08(
    w: Field[[CellDim, KDim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
) -> tuple[Field[[CellDim, KDim], float], Field[[CellDim, KDim], float]]:
    dwdx = neighbor_sum(geofac_grg_x * w(C2E2CO), axis=C2E2CODim)
    dwdy = neighbor_sum(geofac_grg_y * w(C2E2CO), axis=C2E2CODim)
    return dwdx, dwdy


@field_operator
def _mo_nh_diffusion_stencil_08_dwdx(
    w: Field[[CellDim, KDim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_nh_diffusion_stencil_08(w, geofac_grg_x, geofac_grg_y)[0]


@field_operator
def _mo_nh_diffusion_stencil_08_dwdy(
    w: Field[[CellDim, KDim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
) -> Field[[CellDim, KDim], float]:
    return _mo_nh_diffusion_stencil_08(w, geofac_grg_x, geofac_grg_y)[1]


@program
def mo_nh_diffusion_stencil_08(
    w: Field[[CellDim, KDim], float],
    geofac_grg_x: Field[[CellDim, C2E2CODim], float],
    geofac_grg_y: Field[[CellDim, C2E2CODim], float],
    dwdx: Field[[CellDim, KDim], float],
    dwdy: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_08_dwdx(w, geofac_grg_x, geofac_grg_y, out=dwdx)
    _mo_nh_diffusion_stencil_08_dwdy(w, geofac_grg_x, geofac_grg_y, out=dwdy)
