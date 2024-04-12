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
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import Field, astype, int32, where

from icon4py.model.common.dimension import C2CEC, C2E2C, CECDim, CellDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], wpfloat],
    geofac_n2s_c: Field[[CellDim], wpfloat],
    geofac_n2s_nbh: Field[[CECDim], wpfloat],
    vcoef: Field[[CECDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    z_temp: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_temp_wp = astype(z_temp, wpfloat)

    theta_v_0 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[0])))
    theta_v_1 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[1])))
    theta_v_2 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[2])))

    theta_v_0_m1 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[0]) + 1))
    theta_v_1_m1 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[1]) + 1))
    theta_v_2_m1 = theta_v(as_offset(Koff, zd_vertoffset(C2CEC[2]) + 1))

    sum_tmp = (
        theta_v * geofac_n2s_c
        + geofac_n2s_nbh(C2CEC[0])
        * (
            vcoef(C2CEC[0]) * theta_v_0(C2E2C[0])
            + (wpfloat("1.0") - vcoef(C2CEC[0])) * theta_v_0_m1(C2E2C[0])
        )
        + geofac_n2s_nbh(C2CEC[1])
        * (
            vcoef(C2CEC[1]) * theta_v_1(C2E2C[1])
            + (wpfloat("1.0") - vcoef(C2CEC[1])) * theta_v_1_m1(C2E2C[1])
        )
        + geofac_n2s_nbh(C2CEC[2])
        * (
            vcoef(C2CEC[2]) * theta_v_2(C2E2C[2])
            + (wpfloat("1.0") - vcoef(C2CEC[2])) * theta_v_2_m1(C2E2C[2])
        )
    )

    z_temp_wp = where(
        mask,
        z_temp_wp + zd_diffcoef * sum_tmp,
        z_temp_wp,
    )

    return astype(z_temp_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: Field[[CellDim, KDim], bool],
    zd_vertoffset: Field[[CECDim, KDim], int32],
    zd_diffcoef: Field[[CellDim, KDim], wpfloat],
    geofac_n2s_c: Field[[CellDim], wpfloat],
    geofac_n2s_nbh: Field[[CECDim], wpfloat],
    vcoef: Field[[CECDim, KDim], wpfloat],
    theta_v: Field[[CellDim, KDim], wpfloat],
    z_temp: Field[[CellDim, KDim], vpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
        mask,
        zd_vertoffset,
        zd_diffcoef,
        geofac_n2s_c,
        geofac_n2s_nbh,
        vcoef,
        theta_v,
        z_temp,
        out=z_temp,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
