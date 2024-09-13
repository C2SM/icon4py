# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import GridType
from gt4py.next.ffront.decorator import field_operator, program
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff
from icon4py.model.common.type_alias import wpfloat


@field_operator
def _integrate_tracer_vertically_a(
    tracer_now: fa.CellKField[wpfloat],
    rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    rhodz_new: fa.CellKField[wpfloat],
    p_dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    tracer_new = (
        tracer_now * rhodz_now
        + p_dtime * (p_mflx_tracer_v(Koff[1]) * deepatmo_divzl - p_mflx_tracer_v * deepatmo_divzu)
    ) / rhodz_new

    return tracer_new


@field_operator
def _integrate_tracer_vertically(
    tracer_now: fa.CellKField[wpfloat],
    rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    rhodz_new: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    p_dtime: wpfloat,
    ivadv_tracer: int32,
    iadv_slev_jt: int32,
) -> fa.CellKField[wpfloat]:
    k = broadcast(k, (CellDim, KDim))

    tracer_new = (
        where(
            (iadv_slev_jt <= k),
            _integrate_tracer_vertically_a(
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
def integrate_tracer_vertically(
    tracer_now: fa.CellKField[wpfloat],
    rhodz_now: fa.CellKField[wpfloat],
    p_mflx_tracer_v: fa.CellKField[wpfloat],
    deepatmo_divzl: fa.KField[wpfloat],
    deepatmo_divzu: fa.KField[wpfloat],
    rhodz_new: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    p_dtime: wpfloat,
    ivadv_tracer: int32,
    iadv_slev_jt: int32,
    tracer_new: fa.CellKField[wpfloat],
):
    _integrate_tracer_vertically(
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
