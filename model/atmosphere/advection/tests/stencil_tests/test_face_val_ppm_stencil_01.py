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
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_01 import face_val_ppm_stencil_01
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import _shape, random_field


def face_val_ppm_stencil_01_numpy(
    p_cc: np.array,
    p_cellhgt_mc_now: np.array,
    k: np.array,
    elev: int32,
):
    # this is a comment: k = np.broadcast_to(k, p_cc.shape)

    # 01a
    zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
        p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
    )
    zfac = (p_cc[:, 2:] - p_cc[:, 1:-1]) / (p_cellhgt_mc_now[:, 2:] + p_cellhgt_mc_now[:, 1:-1])
    z_slope_a = (
        p_cellhgt_mc_now[:, 1:-1]
        / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 2:])
    ) * (
        (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
        + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 2:]) * zfac_m1
    )

    # 01b
    zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
        p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
    )
    zfac = (p_cc[:, 1:-1] - p_cc[:, 1:-1]) / (p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1])
    z_slope_b = (
        p_cellhgt_mc_now[:, 1:-1]
        / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1])
    ) * (
        (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
        + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 1:-1]) * zfac_m1
    )

    z_slope = np.where(k[1:-1] < elev, z_slope_a, z_slope_b)

    return z_slope


def test_face_val_ppm_stencil_01(backend):
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim, extend={KDim: 1})
    p_cellhgt_mc_now = random_field(grid, CellDim, KDim, extend={KDim: 1})

    k = as_field((KDim,), np.arange(0, _shape(grid, KDim, extend={KDim: 1})[0], dtype=int32))
    elev = k[-2]

    z_slope = random_field(grid, CellDim, KDim)

    ref = face_val_ppm_stencil_01_numpy(
        p_cc.asnumpy(),
        p_cellhgt_mc_now.asnumpy(),
        k.asnumpy(),
        elev,
    )

    face_val_ppm_stencil_01.with_backend(backend)(
        p_cc,
        p_cellhgt_mc_now,
        k,
        elev,
        z_slope,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref[:, :-1], z_slope.asnumpy()[:, 1:-1])
