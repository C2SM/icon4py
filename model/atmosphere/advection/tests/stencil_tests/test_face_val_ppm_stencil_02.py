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
from gt4py.next.ffront.fbuiltins import int32
from gt4py.next.iterator import embedded as it_embedded

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02 import face_val_ppm_stencil_02
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.grid.simple import SimpleGrid
from icon4py.model.common.test_utils.helpers import _shape, random_field


def face_val_ppm_stencil_02_numpy(
    p_cc: np.array,
    p_cellhgt_mc_now: np.array,
    p_face_in: np.array,
    k: np.array,
    slev: int32,
    elev: int32,
    slevp1: int32,
    elevp1: int32,
):
    p_face_a = p_face_in

    p_face_a[:, 1:] = p_cc[:, 1:] * (1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])) + (
        p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])
    ) * ((p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1])

    p_face = np.where((k == slevp1) | (k == elev), p_face_a, p_face_in)
    p_face = np.where((k == slev), p_cc, p_face)
    p_face[:, 1:] = np.where((k[1:] == elevp1), p_cc[:, :-1], p_face[:, 1:])

    return p_face


def test_face_val_ppm_stencil_02():
    grid = SimpleGrid()
    p_cc = random_field(grid, CellDim, KDim)
    p_cellhgt_mc_now = random_field(grid, CellDim, KDim)
    p_face_in = random_field(grid, CellDim, KDim)
    p_face = random_field(grid, CellDim, KDim)

    k = it_embedded.np_as_located_field(KDim)(np.arange(0, _shape(grid, KDim)[0], dtype=int32))

    slev = int32(1)
    slevp1 = slev + int32(1)
    elev = k[-3]
    elevp1 = elev + int32(1)

    ref = face_val_ppm_stencil_02_numpy(
        np.asarray(p_cc),
        np.asarray(p_cellhgt_mc_now),
        np.asarray(p_face_in),
        np.asarray(k),
        slev,
        elev,
        slevp1,
        elevp1,
    )

    face_val_ppm_stencil_02(
        p_cc,
        p_cellhgt_mc_now,
        p_face_in,
        k,
        slev,
        elev,
        slevp1,
        elevp1,
        p_face,
        offset_provider={"Koff": KDim},
    )

    assert np.allclose(ref, p_face)
