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
from gt4py.next.ffront.fbuiltins import astype, int32
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _add_temporal_tendencies_to_vn(
    vn_nnow: fa.EKwpField,
    ddt_vn_apc_ntl1: fa.EKvpField,
    ddt_vn_phy: fa.EKvpField,
    z_theta_v_e: fa.EKwpField,
    z_gradh_exner: fa.EKvpField,
    dtime: wpfloat,
    cpd: wpfloat,
) -> fa.EKwpField:
    """Formerly known as _mo_solve_nonhydro_stencil_24."""
    z_gradh_exner_wp = astype(z_gradh_exner, wpfloat)

    vn_nnew_wp = vn_nnow + dtime * (
        astype(ddt_vn_apc_ntl1, wpfloat)
        - cpd * z_theta_v_e * z_gradh_exner_wp
        + astype(ddt_vn_phy, wpfloat)
    )
    return vn_nnew_wp


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def add_temporal_tendencies_to_vn(
    vn_nnow: fa.EKwpField,
    ddt_vn_apc_ntl1: fa.EKvpField,
    ddt_vn_phy: fa.EKvpField,
    z_theta_v_e: fa.EKwpField,
    z_gradh_exner: fa.EKvpField,
    vn_nnew: fa.EKwpField,
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _add_temporal_tendencies_to_vn(
        vn_nnow,
        ddt_vn_apc_ntl1,
        ddt_vn_phy,
        z_theta_v_e,
        z_gradh_exner,
        dtime,
        cpd,
        out=vn_nnew,
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
