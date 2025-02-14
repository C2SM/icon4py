# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.experimental import as_offset
from gt4py.next.ffront.fbuiltins import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2CEC, C2E2C, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: fa.CellKField[bool],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CECDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CECDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CECDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = astype(z_temp, wpfloat)

    theta_v_0 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset(C2CEC[0])))
    theta_v_1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset(C2CEC[1])))
    theta_v_2 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset(C2CEC[2])))

    theta_v_0_m1 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset(C2CEC[0]) + 1))
    theta_v_1_m1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset(C2CEC[1]) + 1))
    theta_v_2_m1 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset(C2CEC[2]) + 1))

    sum_tmp = (
        theta_v * geofac_n2s_c
        + geofac_n2s_nbh(C2CEC[0])
        * (vcoef(C2CEC[0]) * theta_v_0 + (wpfloat("1.0") - vcoef(C2CEC[0])) * theta_v_0_m1)
        + geofac_n2s_nbh(C2CEC[1])
        * (vcoef(C2CEC[1]) * theta_v_1 + (wpfloat("1.0") - vcoef(C2CEC[1])) * theta_v_1_m1)
        + geofac_n2s_nbh(C2CEC[2])
        * (vcoef(C2CEC[2]) * theta_v_2 + (wpfloat("1.0") - vcoef(C2CEC[2])) * theta_v_2_m1)
    )

    z_temp_wp = where(
        mask,
        z_temp_wp + zd_diffcoef * sum_tmp,
        z_temp_wp,
    )

    return astype(z_temp_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    mask: fa.CellKField[bool],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CECDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CECDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CECDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
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
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
