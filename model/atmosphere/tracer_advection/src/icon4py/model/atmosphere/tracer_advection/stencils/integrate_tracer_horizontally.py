# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E


@gtx.field_operator
def _integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    tracer_new_hor = (
        tracer_now * rhodz_now
        - p_dtime
        * deepatmo_divh
        * neighbor_sum(p_mflx_tracer_h(C2E) * geofac_div, axis=dims.C2EDim)
    ) / rhodz_new

    return tracer_new_hor


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def integrate_tracer_horizontally(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    tracer_now: fa.CellKField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    tracer_new_hor: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _integrate_tracer_horizontally(
        p_mflx_tracer_h=p_mflx_tracer_h,
        deepatmo_divh=deepatmo_divh,
        tracer_now=tracer_now,
        rhodz_now=rhodz_now,
        rhodz_new=rhodz_new,
        geofac_div=geofac_div,
        p_dtime=p_dtime,
        out=tracer_new_hor,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _integrate_tracers_horizontally(
    # pairs rather than two tuples: there is no `zip` inside a field operator
    fluxes_and_tracers: tuple[tuple[fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat]], ...],
    deepatmo_divh: fa.KField[ta.wpfloat],
    rhodz_now: fa.CellKField[ta.wpfloat],
    rhodz_new: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], ...]:
    return tuple(
        _integrate_tracer_horizontally(
            p_mflx_tracer_h, deepatmo_divh, tracer_now, rhodz_now, rhodz_new, geofac_div, p_dtime
        )
        for p_mflx_tracer_h, tracer_now in fluxes_and_tracers
    )


@functools.cache
def integrate_tracers_horizontally(ntracer: int) -> gtx_typing.Program:
    # A program with a variable-length tuple parameter cannot be compiled ahead of time
    # (`setup_program`), so the tuple length is fixed per program.
    tracers = tuple[(fa.CellKField[ta.wpfloat],) * ntracer]  # type: ignore[misc]
    fluxes_and_tracers = tuple[  # type: ignore[misc]
        (tuple[fa.EdgeKField[ta.wpfloat], fa.CellKField[ta.wpfloat]],) * ntracer
    ]

    @gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
    def integrate_tracers_horizontally(
        fluxes_and_tracers: fluxes_and_tracers,
        deepatmo_divh: fa.KField[ta.wpfloat],
        rhodz_now: fa.CellKField[ta.wpfloat],
        rhodz_new: fa.CellKField[ta.wpfloat],
        geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
        tracers_new_hor: tracers,
        p_dtime: ta.wpfloat,
        horizontal_start: gtx.int32,
        horizontal_end: gtx.int32,
        vertical_start: gtx.int32,
        vertical_end: gtx.int32,
    ) -> None:
        _integrate_tracers_horizontally(
            fluxes_and_tracers=fluxes_and_tracers,
            deepatmo_divh=deepatmo_divh,
            rhodz_now=rhodz_now,
            rhodz_new=rhodz_new,
            geofac_div=geofac_div,
            p_dtime=p_dtime,
            out=tracers_new_hor,
            domain={
                dims.CellDim: (horizontal_start, horizontal_end),
                dims.KDim: (vertical_start, vertical_end),
            },
        )

    return integrate_tracers_horizontally
