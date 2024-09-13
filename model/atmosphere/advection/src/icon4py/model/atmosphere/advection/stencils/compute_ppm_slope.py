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

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import Koff
from icon4py.model.common.type_alias import wpfloat


# TODO: this will have to be removed once domain allows for imports
CellDim = dims.CellDim
KDim = dims.KDim


@field_operator
def _compute_ppm_slope_a(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    zfac_m1 = (p_cc - p_cc(Koff[-1])) / (p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1]))
    zfac = (p_cc(Koff[+1]) - p_cc) / (p_cellhgt_mc_now(Koff[+1]) + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now
        / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[+1]))
    ) * (
        (wpfloat(2.0) * p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + wpfloat(2.0) * p_cellhgt_mc_now(Koff[+1])) * zfac_m1
    )

    return z_slope


@field_operator
def _compute_ppm_slope_b(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    zfac_m1 = (p_cc - p_cc(Koff[-1])) / (p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1]))
    zfac = (p_cc - p_cc) / (p_cellhgt_mc_now + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now + p_cellhgt_mc_now)
    ) * (
        (wpfloat(2.0) * p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + wpfloat(2.0) * p_cellhgt_mc_now) * zfac_m1
    )

    return z_slope


@field_operator
def _compute_ppm_slope(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    elev: int32,
) -> fa.CellKField[wpfloat]:
    k = broadcast(k, (CellDim, KDim))

    z_slope = where(
        k == elev,
        _compute_ppm_slope_b(p_cc, p_cellhgt_mc_now),
        _compute_ppm_slope_a(p_cc, p_cellhgt_mc_now),
    )

    return z_slope


@program(grid_type=GridType.UNSTRUCTURED)
def compute_ppm_slope(
    p_cc: fa.CellKField[wpfloat],
    p_cellhgt_mc_now: fa.CellKField[wpfloat],
    k: fa.KField[int32],
    elev: int32,
    z_slope: fa.CellKField[wpfloat],
    horizontal_start: int32,
    horizontal_end: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_ppm_slope(
        p_cc,
        p_cellhgt_mc_now,
        k,
        elev,
        out=z_slope,
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
