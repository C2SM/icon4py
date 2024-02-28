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
import pytest
from gt4py.next import as_field
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.advection.face_val_ppm_stencil_02 import face_val_ppm_stencil_02
from icon4py.model.common.dimension import CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, _shape, random_field, zero_field


class TestFaceValPpmStencil02(StencilTest):
    PROGRAM = face_val_ppm_stencil_02
    OUTPUTS = ("p_face",)

    @staticmethod
    def reference(
        grid,
        p_cc: np.array,
        p_cellhgt_mc_now: np.array,
        p_face_in: np.array,
        k: np.array,
        slev: int32,
        elev: int32,
        slevp1: int32,
        elevp1: int32,
        **kwargs,
    ):
        p_face_a = p_face_in
        p_face_a[:, 1:] = p_cc[:, 1:] * (
            1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])
        ) + (p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])) * (
            (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1]
        )

        p_face = np.where((k == slevp1) | (k == elev), p_face_a, p_face_in)
        p_face = np.where((k == slev), p_cc, p_face)
        p_face[:, 1:] = np.where((k[1:] == elevp1), p_cc[:, :-1], p_face[:, 1:])
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid):
        p_cc = random_field(grid, CellDim, KDim)
        p_cellhgt_mc_now = random_field(grid, CellDim, KDim)
        p_face_in = random_field(grid, CellDim, KDim)
        p_face = zero_field(grid, CellDim, KDim)

        k = as_field((KDim,), np.arange(0, _shape(grid, KDim)[0], dtype=int32))
        slev = int32(1)
        slevp1 = slev + int32(1)
        elev = int32(k[-3].as_scalar())
        elevp1 = elev + int32(1)

        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            p_face_in=p_face_in,
            k=k,
            slev=slev,
            elev=elev,
            slevp1=slevp1,
            elevp1=elevp1,
            p_face=p_face,
        )
