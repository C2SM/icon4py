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

from icon4py.model.atmosphere.dycore.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim, Koff
from icon4py.model.common.settings import backend
from icon4py.model.common.type_alias import vpfloat, wpfloat


@field_operator
def _interpolate_vn_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EKvpField,
    vn: fa.EKwpField,
    vt: fa.EKvpField,
) -> tuple[fa.EKvpField, fa.EKvpField]:
    """Formerly known as _mo_velocity_advection_stencil_02."""
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)

    vn_ie_wp = wgtfac_e_wp * vn + (wpfloat("1.0") - wgtfac_e_wp) * vn(Koff[-1])
    _, _, z_kin_hor_e = _compute_horizontal_kinetic_energy(vn=vn, vt=vt)

    return astype(vn_ie_wp, vpfloat), z_kin_hor_e


@program(grid_type=GridType.UNSTRUCTURED, backend=backend)
def interpolate_vn_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EKvpField,
    vn: fa.EKwpField,
    vt: fa.EKvpField,
    vn_ie: fa.EKvpField,
    z_kin_hor_e: fa.EKvpField,
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _interpolate_vn_to_ie_and_compute_ekin_on_edges(
        wgtfac_e,
        vn,
        vt,
        out=(vn_ie, z_kin_hor_e),
        domain={
            EdgeDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
