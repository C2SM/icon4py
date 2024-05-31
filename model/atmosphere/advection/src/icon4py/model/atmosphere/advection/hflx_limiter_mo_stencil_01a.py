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
from gt4py.next.ffront.fbuiltins import abs

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import E2C


@field_operator
def _hflx_limiter_mo_stencil_01a(
    p_mflx_tracer_h: fa.EKfloatField,
    p_mass_flx_e: fa.EKfloatField,
    p_cc: fa.CKfloatField,
) -> tuple[fa.EKfloatField, fa.EKfloatField]:
    z_mflx_low = 0.5 * (
        p_mass_flx_e * (p_cc(E2C[0]) + p_cc(E2C[1]))
        - abs(p_mass_flx_e) * (p_cc(E2C[1]) - p_cc(E2C[0]))
    )

    z_anti = p_mflx_tracer_h - z_mflx_low

    return (z_mflx_low, z_anti)


@program
def hflx_limiter_mo_stencil_01a(
    p_mflx_tracer_h: fa.EKfloatField,
    p_mass_flx_e: fa.EKfloatField,
    p_cc: fa.CKfloatField,
    z_mflx_low: fa.EKfloatField,
    z_anti: fa.EKfloatField,
):
    _hflx_limiter_mo_stencil_01a(
        p_mflx_tracer_h,
        p_mass_flx_e,
        p_cc,
        out=(z_mflx_low, z_anti),
    )
