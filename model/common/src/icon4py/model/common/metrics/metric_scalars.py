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
import numpy as np
from gt4py.next import Field, broadcast, field_operator, int32, program, where

from icon4py.model.common.dimension import KDim, Koff


@field_operator
def _compute_scalfac_dd3d(
    vct_a: Field[[KDim], float],
    divdamp_trans_start: float,
    divdamp_trans_end: float,
    divdamp_type: int32,
) -> Field[[KDim], float]:
    scalfac_dd3d = broadcast(1.0, (KDim,))
    if divdamp_type == 32:
        zf = 0.5 * (vct_a + vct_a(Koff[1]))
        scalfac_dd3d = where(zf >= divdamp_trans_end, 0.0, scalfac_dd3d)
        scalfac_dd3d = where(
            zf >= divdamp_trans_start,
            (divdamp_trans_end - zf) / (divdamp_trans_end - divdamp_trans_start),
            scalfac_dd3d,
        )
    return scalfac_dd3d


@program
def compute_scalfac_dd3d(
    vct_a: Field[[KDim], float],
    scalfac_dd3d: Field[[KDim], float],
    divdamp_trans_start: float,
    divdamp_trans_end: float,
    divdamp_type: int32,
    vertical_start: int32,
    vertical_end: int32,
):
    _compute_scalfac_dd3d(
        vct_a,
        divdamp_trans_start,
        divdamp_trans_end,
        divdamp_type,
        out=scalfac_dd3d,
        domain={KDim: (vertical_start, vertical_end)},
    )


def compute_kstart_dd3d(
    vct_a: np.array, k_levels: np.array, divdamp_trans_end: float, divdamp_type: int
) -> int:
    kstart_dd3d = 1
    kstart_dd3d_k = 1
    if divdamp_type == 32:
        zf = 0.5 * (vct_a[: k_levels.size] + vct_a[1:])  # Koff[1])
        kstart_dd3d_k_arr = np.where(zf >= divdamp_trans_end, k_levels, kstart_dd3d)
        kstart_dd3d_ls = list((i for i, x in enumerate(kstart_dd3d_k_arr) if x != 2))
        if len(kstart_dd3d_ls) > 0:
            kstart_dd3d_k = kstart_dd3d_k_arr[kstart_dd3d_ls[0]]
        else:
            kstart_dd3d_k = kstart_dd3d_k_arr[0]
    kstart_dd3d = kstart_dd3d_k + 1
    return kstart_dd3d
