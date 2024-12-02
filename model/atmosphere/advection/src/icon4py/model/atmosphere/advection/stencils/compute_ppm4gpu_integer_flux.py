# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import abs, astype, floor, where

from icon4py.model.atmosphere.advection.stencils.compute_ppm4gpu_fractional_flux import (
    _sum_neighbor_contributions,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): this stencil has no test
# TODO (dastrm): this stencil does not strictly match the fortran code


@gtx.field_operator
def _compute_ppm4gpu_integer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    p_upflux: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    js = floor(abs(z_cfl)) - 1.0

    z_cfl_pos = z_cfl > 0.0
    z_cfl_neg = not z_cfl_pos
    wsign = where(z_cfl_pos, 1.0, -1.0)

    in_slev_bounds = astype(k, wpfloat) - js >= astype(slev, wpfloat)

    p_cc_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, p_cc)
    p_cellmass_now_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, p_cellmass_now)

    z_iflx = wsign * p_cc_jks * p_cellmass_now_jks

    p_upflux = p_upflux + where(in_slev_bounds, z_iflx / p_dtime, 0.0)

    return p_upflux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm4gpu_integer_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    p_upflux: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ppm4gpu_integer_flux(
        p_cc,
        p_cellmass_now,
        z_cfl,
        p_upflux,
        k,
        slev,
        p_dtime,
        out=p_upflux,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
