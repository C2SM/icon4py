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

from icon4py.model.common.dimension import Koff


@field_operator
def _step_advection_stencil_01(
    rhodz_ast: fa.CKfloatField,
    p_mflx_contra_v: fa.CKfloatField,
    deepatmo_divzl: fa.KfloatField,
    deepatmo_divzu: fa.KfloatField,
    p_dtime: float,
) -> fa.CKfloatField:
    k_offset_up_low = p_dtime * (
        p_mflx_contra_v(Koff[1]) * deepatmo_divzl - p_mflx_contra_v * deepatmo_divzu
    )
    return rhodz_ast + k_offset_up_low


@program(grid_type=GridType.UNSTRUCTURED)
def step_advection_stencil_01(
    rhodz_ast: fa.CKfloatField,
    p_mflx_contra_v: fa.CKfloatField,
    deepatmo_divzl: fa.KfloatField,
    deepatmo_divzu: fa.KfloatField,
    p_dtime: float,
    rhodz_ast2: fa.CKfloatField,
):
    _step_advection_stencil_01(
        rhodz_ast,
        p_mflx_contra_v,
        deepatmo_divzl,
        deepatmo_divzu,
        p_dtime,
        out=rhodz_ast2,
    )
