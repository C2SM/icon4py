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
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import Field, int32, where

from icon4py.common.dimension import C2CEC, C2E2C, CECDim, CellDim, KDim, Koff


@field_operator
def step(
    zd_vertoffset: Field[[CECDim, KDim], int32],
    geofac_n2s_nbh: Field[[CECDim], float],
    vcoef: Field[[CECDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    i: int,
) -> Field[[CellDim, KDim], float]:

    theta_v_deref = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[i])))
    theta_v_deref_m1 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[i]) + int32(1)))

    return geofac_n2s_nbh(C2CEC[i]) *(vcoef(C2CEC[i]) * theta_v_deref(C2E2C[i]) + (1.0 - vcoef(C2CEC[i])) * theta_v_deref_m1(C2E2C[i]))

@field_operator
def _mo_nh_diffusion_stencil_15(
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CECDim], float],
    vcoef: Field[[CECDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
) -> Field[[CellDim, KDim], float]:

    sum = step(zd_vertoffset, geofac_n2s_nbh, vcoef, theta_v, 0) + step(zd_vertoffset, geofac_n2s_nbh, vcoef, theta_v, 1) + step(zd_vertoffset, geofac_n2s_nbh, vcoef, theta_v, 2)

    z_temp = where(mask, z_temp + zd_diffcoef * (theta_v * geofac_n2s_c + sum), z_temp)

    return z_temp


@program
def mo_nh_diffusion_stencil_15(
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], float],
    geofac_n2s_c: Field[[CellDim], float],
    geofac_n2s_nbh: Field[[CECDim], float],
    vcoef: Field[[CECDim, KDim], float],
    theta_v: Field[[CellDim, KDim], float],
    z_temp: Field[[CellDim, KDim], float],
):
    _mo_nh_diffusion_stencil_15(
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v,
        z_temp,
        out=z_temp,
    )
