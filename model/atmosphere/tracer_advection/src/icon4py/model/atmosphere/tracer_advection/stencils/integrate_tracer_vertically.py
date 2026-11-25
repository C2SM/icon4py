# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.tracer_advection.tracer_advection_type import VerticalAdvectionType
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


# TODO(nfarabullini, OngChia): k/iadv_slev_jt and vertical_start/end are redundant


@gtx.field_operator
def _integrate_tracer_vertically_a(
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    tracer_new = (
        tracer_now * rhodz_now
        + p_dtime
        * (
            p_mflx_tracer_v(dims.KDim + 0.5) * deepatmo_divzl
            - p_mflx_tracer_v(dims.KDim - 0.5) * deepatmo_divzu
        )
    ) / rhodz_new

    return tracer_new


@gtx.field_operator
def _integrate_tracer_vertically(
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    tracer_new = (
        concat_where(
            dims.KDim >= iadv_slev_jt,
            _integrate_tracer_vertically_a(
                tracer_now=tracer_now,
                rhodz_now=rhodz_now,
                p_mflx_tracer_v=p_mflx_tracer_v,
                deepatmo_divzl=deepatmo_divzl,
                deepatmo_divzu=deepatmo_divzu,
                rhodz_new=rhodz_new,
                p_dtime=p_dtime,
            ),
            tracer_now,
        )
        if (ivadv_tracer != VerticalAdvectionType.NO_ADVECTION)
        else tracer_now
    )

    return tracer_new


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def integrate_tracer_vertically(
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_v: fa.CellKHalfField[ta.wpfloat],
    deepatmo_divzl: fa.KField[ta.wpfloat],
    deepatmo_divzu: fa.KField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    tracer_new: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    ivadv_tracer: gtx.int32,
    iadv_slev_jt: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _integrate_tracer_vertically(
        tracer_now=tracer_now,
        rhodz_now=rhodz_now,
        p_mflx_tracer_v=p_mflx_tracer_v,
        deepatmo_divzl=deepatmo_divzl,
        deepatmo_divzu=deepatmo_divzu,
        rhodz_new=rhodz_new,
        p_dtime=p_dtime,
        ivadv_tracer=ivadv_tracer,
        iadv_slev_jt=iadv_slev_jt,
        out=tracer_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
