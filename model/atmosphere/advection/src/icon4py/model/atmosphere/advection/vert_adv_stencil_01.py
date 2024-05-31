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
from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import Field, broadcast, int32, where
from model.common.tests import field_type_aliases as fa

from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _vert_adv_stencil_01a(
    tracer_now: fa.CKfloatField,
    rhodz_now: fa.CKfloatField,
    p_mflx_tracer_v: fa.CKfloatField,
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    rhodz_new: fa.CKfloatField,
    p_dtime: float,
) -> fa.CKfloatField:
    tracer_new = (
        tracer_now * rhodz_now
        + p_dtime * (p_mflx_tracer_v(Koff[1]) * deepatmo_divzl - p_mflx_tracer_v * deepatmo_divzu)
    ) / rhodz_new

    return tracer_new


@field_operator
def _vert_adv_stencil_01(
    tracer_now: fa.CKfloatField,
    rhodz_now: fa.CKfloatField,
    p_mflx_tracer_v: fa.CKfloatField,
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    rhodz_new: fa.CKfloatField,
    k: fa.KintField,
    p_dtime: float,
    ivadv_tracer: int32,
    iadv_slev_jt: int32,
) -> fa.CKfloatField:
    k = broadcast(k, (CellDim, KDim))

    tracer_new = (
        where(
            (iadv_slev_jt <= k),
            _vert_adv_stencil_01a(
                tracer_now,
                rhodz_now,
                p_mflx_tracer_v,
                deepatmo_divzl,
                deepatmo_divzu,
                rhodz_new,
                p_dtime,
            ),
            tracer_now,
        )
        if (ivadv_tracer != 0)
        else tracer_now
    )

    return tracer_new


@program(grid_type=GridType.UNSTRUCTURED)
def vert_adv_stencil_01(
    tracer_now: fa.CKfloatField,
    rhodz_now: fa.CKfloatField,
    p_mflx_tracer_v: fa.CKfloatField,
    deepatmo_divzl: Field[[KDim], float],
    deepatmo_divzu: Field[[KDim], float],
    rhodz_new: fa.CKfloatField,
    k: fa.KintField,
    p_dtime: float,
    ivadv_tracer: int32,
    iadv_slev_jt: int32,
    tracer_new: fa.CKfloatField,
):
    _vert_adv_stencil_01(
        tracer_now,
        rhodz_now,
        p_mflx_tracer_v,
        deepatmo_divzl,
        deepatmo_divzu,
        rhodz_new,
        k,
        p_dtime,
        ivadv_tracer,
        iadv_slev_jt,
        out=tracer_new,
    )
