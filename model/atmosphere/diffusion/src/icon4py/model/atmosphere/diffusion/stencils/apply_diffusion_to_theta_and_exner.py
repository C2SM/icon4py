# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, where
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, Koff
from icon4py.model.common.math.differential_operators import divergence, normal_gradient
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = astype(z_temp, wpfloat)

    theta_v_0 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(0)]))
    theta_v_1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(1)]))
    theta_v_2 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset[dims.C2E2CDim(2)]))

    # `zd_vertoffset` is 0 where `zd_diffcoef` is 0, so there the `+ 1` target at the bottom level
    # lies below the column; keep the read in range for backends that evaluate every point.
    is_steep = zd_diffcoef != wpfloat("0.0")
    theta_v_0_m1 = theta_v(C2E2C[0])(
        as_offset(Koff, where(is_steep, zd_vertoffset[dims.C2E2CDim(0)] + 1, 0))
    )
    theta_v_1_m1 = theta_v(C2E2C[1])(
        as_offset(Koff, where(is_steep, zd_vertoffset[dims.C2E2CDim(1)] + 1, 0))
    )
    theta_v_2_m1 = theta_v(C2E2C[2])(
        as_offset(Koff, where(is_steep, zd_vertoffset[dims.C2E2CDim(2)] + 1, 0))
    )

    sum_tmp = (
        theta_v * geofac_n2s_c
        + geofac_n2s_nbh[dims.C2E2CDim(0)]
        * (
            vcoef[dims.C2E2CDim(0)] * theta_v_0
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(0)]) * theta_v_0_m1
        )
        + geofac_n2s_nbh[dims.C2E2CDim(1)]
        * (
            vcoef[dims.C2E2CDim(1)] * theta_v_1
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(1)]) * theta_v_1_m1
        )
        + geofac_n2s_nbh[dims.C2E2CDim(2)]
        * (
            vcoef[dims.C2E2CDim(2)] * theta_v_2
            + (wpfloat("1.0") - vcoef[dims.C2E2CDim(2)]) * theta_v_2_m1
        )
    )

    # Note: In the original Fortran code `zd_diffcoef` is implemented as a list,
    # in ICON4Py it's a full field intialized with zeros for points that are not in the list.
    z_temp_wp = z_temp_wp + zd_diffcoef * sum_tmp

    return astype(z_temp_wp, vpfloat)


@gtx.field_operator
def _update_theta_and_exner(
    z_temp: fa.CellKField[vpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    rd_o_cvd_wp, z_temp_wp = astype((rd_o_cvd, z_temp), wpfloat)

    z_theta = theta_v
    theta_v = theta_v + (area * z_temp_wp)
    exner = exner * (wpfloat("1.0") + rd_o_cvd_wp * (theta_v / z_theta - wpfloat("1.0")))
    return theta_v, exner


@gtx.field_operator
def _apply_diffusion_to_theta_and_exner(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    area: fa.CellField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    grad_temp = astype(kh_smag_e, wpfloat) * normal_gradient(theta_v, inv_dual_edge_length)
    z_temp = astype(divergence(grad_temp, geofac_div), vpfloat)

    if apply_zdiffusion_t:
        # Only where `zd_diffcoef` is not 0 to avoid loading the other fields.
        # Eventually the toolchain could consider extracting a mask `zd_diffcoef != 0` if we tell gt4py that `zd_diffcoef` is static.
        z_temp = where(
            zd_diffcoef != wpfloat(0.0),
            _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
                zd_vertoffset,
                zd_diffcoef,
                geofac_n2s_c,
                geofac_n2s_nbh,
                vcoef,
                theta_v,
                z_temp,
            ),
            z_temp,
        )

    theta_v, exner = _update_theta_and_exner(z_temp, area, theta_v, exner, rd_o_cvd)

    return theta_v, exner
