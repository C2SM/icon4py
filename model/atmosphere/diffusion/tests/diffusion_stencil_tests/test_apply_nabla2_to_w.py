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
from gt4py.next.ffront.fbuiltins import int32

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import apply_nabla2_to_w
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


def apply_nabla2_to_w_numpy(
    mesh,
    area: np.array,
    z_nabla2_c: np.array,
    geofac_n2s: np.array,
    w: np.array,
    diff_multfac_w: float,
)-> np.array:
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    area = np.expand_dims(area, axis=-1)
    w = w - diff_multfac_w * area * area * np.sum(z_nabla2_c[mesh.c2e2cO] * geofac_n2s, axis=1)
    return w


class TestMoApplyNabla2ToW(StencilTest):
    PROGRAM = apply_nabla2_to_w
    OUTPUTS = ("w",)

    @staticmethod
    def reference(
        mesh,
        area: np.array,
        z_nabla2_c: np.array,
        geofac_n2s: np.array,
        w: np.array,
        diff_multfac_w: float,
        **kwargs,
    ) -> dict:
        w = apply_nabla2_to_w_numpy(mesh, area, z_nabla2_c, geofac_n2s, w, diff_multfac_w)
        return dict(w=w)

    @pytest.fixture
    def input_data(self, mesh):
        area = random_field(mesh, CellDim)
        z_nabla2_c = random_field(mesh, CellDim, KDim)
        geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
        w = random_field(mesh, CellDim, KDim)
        return dict(
            area=area,
            z_nabla2_c=z_nabla2_c,
            geofac_n2s=geofac_n2s,
            w=w,
            diff_multfac_w=5.0,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_cells),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
