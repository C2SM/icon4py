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

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    calculate_nabla2_for_w,
)
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


def calculate_nabla2_for_w_numpy(mesh, w: np.array, geofac_n2s: np.array):
    geofac_n2s = np.expand_dims(geofac_n2s, axis=-1)
    z_nabla2_c = np.sum(w[mesh.c2e2cO] * geofac_n2s, axis=1)
    return z_nabla2_c


class TestCalculateNabla2ForW(StencilTest):
    PROGRAM = calculate_nabla2_for_w
    OUTPUTS = ("z_nabla2_c",)

    @staticmethod
    def reference(mesh, w: np.array, geofac_n2s: np.array, **kwargs) -> np.array:
        z_nabla2_c = calculate_nabla2_for_w_numpy(mesh, w, geofac_n2s)
        return dict(z_nabla2_c=z_nabla2_c)

    @pytest.fixture
    def input_data(self, mesh):
        w = random_field(mesh, CellDim, KDim)
        geofac_n2s = random_field(mesh, CellDim, C2E2CODim)
        z_nabla2_c = zero_field(mesh, CellDim, KDim)

        return dict(
            w=w,
            geofac_n2s=geofac_n2s,
            z_nabla2_c=z_nabla2_c,
            horizontal_start=int32(0),
            horizontal_end=int32(mesh.n_cells),
            vertical_start=int32(0),
            vertical_end=int32(mesh.k_level),
        )
