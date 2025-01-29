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
from gt4py.next.ffront.fbuiltins import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
) -> tuple[
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
    fa.EdgeKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_37 or _mo_velocity_advection_stencil_05."""
    # TODO: This stencil doesn't only do what the name implies. It also
    # assigns to vn_ie_wp and z_vt_ie_vp. These things should be separated.

    # scidoc:
    # Outputs:
    #  - z_w_concorr_me :
    #     $$
    #     \wcc{\n}{\e}{\k} = \vn{\n}{\e}{\k} \pdxn{z} + \vt{\n}{\e}{\k} \pdxt{z}, \quad \k \in [\nflatlev, \nlev)
    #     $$
    #     Compute the contravariant correction to the vertical wind due to
    #     terrain-following coordinate. $\pdxn{}$ and $\pdxt{}$ are the
    #     horizontal derivatives along the normal and tangent directions
    #     respectively (eq. 17 in |ICONdycorePaper|).
    #  - vn_ie :
    #     $$
    #     \vn{\n}{\e}{-1/2} = \vn{\n}{\e}{0}
    #     $$
    #     Set the normal wind at model top equal to the normal wind at the
    #     first full level.
    #  - z_vt_ie :
    #     $$
    #     \vt{\n}{\e}{-1/2} = \vt{\n}{\e}{0}
    #     $$
    #     Set the tangential wind at model top equal to the tangential wind
    #     at the first full level.
    #  - z_kin_hor_e :
    #     $$
    #     \kinehori{\n}{\e}{0} = \frac{1}{2} \left( \vn{\n}{\e}{0}^2 + \vt{\n}{\e}{0}^2 \right)
    #     $$
    #     Compute the horizontal kinetic energy on the first full level.
    #
    # Inputs:
    #  - $\vn{\n}{\e}{\k}$ : vn
    #  - $\vt{\n}{\e}{\k}$ : vt
    #  - $\pdxn{z}$ : ddxn_z_full
    #  - $\pdxt{z}$ : ddxt_z_full
    #

    vn_ie_wp = vn
    z_vt_ie_vp = vt
    z_kin_hor_e_wp = wpfloat("0.5") * (vn * vn + astype(vt * vt, wpfloat))
    return astype(vn_ie_wp, vpfloat), z_vt_ie_vp, astype(z_kin_hor_e_wp, vpfloat)


@program(grid_type=GridType.UNSTRUCTURED)
def compute_horizontal_kinetic_energy(
    vn: fa.EdgeKField[wpfloat],
    vt: fa.EdgeKField[vpfloat],
    vn_ie: fa.EdgeKField[vpfloat],
    z_vt_ie: fa.EdgeKField[vpfloat],
    z_kin_hor_e: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_horizontal_kinetic_energy(
        vn,
        vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
