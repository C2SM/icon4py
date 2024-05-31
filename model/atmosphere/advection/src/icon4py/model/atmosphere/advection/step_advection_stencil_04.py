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
from model.common.tests import field_aliases as fa


@field_operator
def _step_advection_stencil_04(
    p_tracer_now: fa.CKfloatField,
    p_tracer_new: fa.CKfloatField,
    p_dtime: float,
) -> fa.CKfloatField:
    opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime
    return opt_ddt_tracer_adv


@program(grid_type=GridType.UNSTRUCTURED)
def step_advection_stencil_04(
    p_tracer_now: fa.CKfloatField,
    p_tracer_new: fa.CKfloatField,
    opt_ddt_tracer_adv: fa.CKfloatField,
    p_dtime: float,
):
    _step_advection_stencil_04(p_tracer_now, p_tracer_new, p_dtime, out=opt_ddt_tracer_adv)
