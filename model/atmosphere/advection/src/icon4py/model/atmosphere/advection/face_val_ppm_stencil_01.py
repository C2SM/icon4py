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
from gt4py.next.ffront.fbuiltins import broadcast, int32, where

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import CellDim, KDim, Koff


@field_operator
def _face_val_ppm_stencil_01a(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
) -> fa.CellKField[float]:
    zfac_m1 = (p_cc - p_cc(Koff[-1])) / (p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1]))
    zfac = (p_cc(Koff[+1]) - p_cc) / (p_cellhgt_mc_now(Koff[+1]) + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now
        / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[+1]))
    ) * (
        (2.0 * p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + 2.0 * p_cellhgt_mc_now(Koff[+1])) * zfac_m1
    )

    return z_slope


@field_operator
def _face_val_ppm_stencil_01b(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
) -> fa.CellKField[float]:
    zfac_m1 = (p_cc - p_cc(Koff[-1])) / (p_cellhgt_mc_now + p_cellhgt_mc_now(Koff[-1]))
    zfac = (p_cc - p_cc) / (p_cellhgt_mc_now + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now / (p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now + p_cellhgt_mc_now)
    ) * (
        (2.0 * p_cellhgt_mc_now(Koff[-1]) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + 2.0 * p_cellhgt_mc_now) * zfac_m1
    )

    return z_slope


@field_operator
def _face_val_ppm_stencil_01(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    k: fa.KField[int32],
    elev: int32,
) -> fa.CellKField[float]:
    k = broadcast(k, (CellDim, KDim))

    z_slope = where(
        k == elev,
        _face_val_ppm_stencil_01b(p_cc, p_cellhgt_mc_now),
        _face_val_ppm_stencil_01a(p_cc, p_cellhgt_mc_now),
    )

    return z_slope


@program(grid_type=GridType.UNSTRUCTURED)
def face_val_ppm_stencil_01(
    p_cc: fa.CellKField[float],
    p_cellhgt_mc_now: fa.CellKField[float],
    k: fa.KField[int32],
    elev: int32,
    z_slope: fa.CellKField[float],
):
    _face_val_ppm_stencil_01(
        p_cc,
        p_cellhgt_mc_now,
        k,
        elev,
        out=z_slope,
    )
