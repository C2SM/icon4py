# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import abs, astype, floor, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


# TODO (dastrm): this stencil has no test
# TODO (dastrm): this stencil does not strictly match the fortran code


@gtx.field_operator
def _sum_neighbor_contributions(
    mask1: fa.CellKField[bool],
    mask2: fa.CellKField[bool],
    js: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    p_cc_p0 = where(mask1 & (js == 0.0), p_cc, 0.0)
    p_cc_p1 = where(mask1 & (js == 1.0), p_cc(Koff[1]), 0.0)
    p_cc_p2 = where(mask1 & (js == 2.0), p_cc(Koff[2]), 0.0)
    p_cc_p3 = where(mask1 & (js == 3.0), p_cc(Koff[3]), 0.0)
    p_cc_p4 = where(mask1 & (js == 4.0), p_cc(Koff[4]), 0.0)
    p_cc_m0 = where(mask2 & (js == 0.0), p_cc(Koff[-1]), 0.0)
    p_cc_m1 = where(mask2 & (js == 1.0), p_cc(Koff[-2]), 0.0)
    p_cc_m2 = where(mask2 & (js == 2.0), p_cc(Koff[-3]), 0.0)
    p_cc_m3 = where(mask2 & (js == 3.0), p_cc(Koff[-4]), 0.0)
    p_cc_m4 = where(mask2 & (js == 4.0), p_cc(Koff[-5]), 0.0)
    p_cc_jks = (
        p_cc_p0
        + p_cc_p1
        + p_cc_p2
        + p_cc_p3
        + p_cc_p4
        + p_cc_m0
        + p_cc_m1
        + p_cc_m2
        + p_cc_m3
        + p_cc_m4
    )
    return p_cc_jks


@gtx.field_operator
def _compute_ppm4gpu_fractional_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    js = floor(abs(z_cfl))
    z_cflfrac = abs(z_cfl) - js

    z_cfl_pos = z_cfl > 0.0
    z_cfl_neg = not z_cfl_pos
    wsign = where(z_cfl_pos, 1.0, -1.0)

    in_slev_bounds = astype(k, wpfloat) - js >= astype(slev, wpfloat)

    p_cc_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, p_cc)
    p_cellmass_now_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, p_cellmass_now)
    z_delta_q_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, z_delta_q)
    z_a1_jks = _sum_neighbor_contributions(z_cfl_pos, z_cfl_neg, js, z_a1)

    z_q_int = (
        p_cc_jks
        + wsign * (z_delta_q_jks * (1.0 - z_cflfrac))
        - z_a1_jks * (1.0 - 3.0 * z_cflfrac + 2.0 * z_cflfrac * z_cflfrac)
    )

    p_upflux = where(
        in_slev_bounds, wsign * p_cellmass_now_jks * z_cflfrac * z_q_int / p_dtime, 0.0
    )

    return p_upflux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm4gpu_fractional_flux(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    p_upflux: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slev: gtx.int32,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _compute_ppm4gpu_fractional_flux(
        p_cc,
        p_cellmass_now,
        z_cfl,
        z_delta_q,
        z_a1,
        k,
        slev,
        p_dtime,
        out=p_upflux,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
