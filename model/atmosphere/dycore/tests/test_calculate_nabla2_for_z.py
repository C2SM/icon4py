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

from icon4py.model.atmosphere.dycore.calculate_nabla2_for_z import calculate_nabla2_for_z
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field


class TestCalculateNabla2ForZ(StencilTest):
    PROGRAM = calculate_nabla2_for_z
    OUTPUTS = ("z_nabla2_e",)

    @staticmethod
    def reference(
        mesh,
        kh_smag_e: np.array,
        inv_dual_edge_length: np.array,
        theta_v: np.array,
        **kwargs,
    ) -> np.array:
        inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

        theta_v_e2c = theta_v[mesh.e2c]
        theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

        z_nabla2_e = kh_smag_e * inv_dual_edge_length * theta_v_weighted
        return dict(z_nabla2_e=z_nabla2_e)

    @pytest.fixture
    def input_data(self, mesh):
        kh_smag_e = random_field(mesh, EdgeDim, KDim)
        inv_dual_edge_length = random_field(mesh, EdgeDim)
        theta_v = random_field(mesh, CellDim, KDim)
        z_nabla2_e = random_field(mesh, EdgeDim, KDim)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v,
            z_nabla2_e=z_nabla2_e,
        )
