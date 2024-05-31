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
from gt4py.next.ffront.fbuiltins import minimum, where
from model.common.tests import field_aliases as fa

from icon4py.model.common.dimension import E2C


@field_operator
def _hflx_limiter_mo_stencil_05(
    z_anti: fa.EKfloatField,
    z_mflx_low: fa.EKfloatField,
    r_m: fa.CKfloatField,
    r_p: fa.CKfloatField,
) -> fa.EKfloatField:
    z_signum = where((z_anti > 0.0), 1.0, -1.0)

    r_frac = 0.5 * (
        (1.0 + z_signum) * minimum(r_m(E2C[0]), r_p(E2C[1]))
        + (1.0 - z_signum) * minimum(r_m(E2C[1]), r_p(E2C[0]))
    )

    p_mflx_tracer_h = z_mflx_low + minimum(1.0, r_frac) * z_anti

    return p_mflx_tracer_h


@program
def hflx_limiter_mo_stencil_05(
    z_anti: fa.EKfloatField,
    z_mflx_low: fa.EKfloatField,
    r_m: fa.CKfloatField,
    r_p: fa.CKfloatField,
    p_mflx_tracer_h: fa.EKfloatField,
):
    _hflx_limiter_mo_stencil_05(z_anti, z_mflx_low, r_m, r_p, out=p_mflx_tracer_h)
