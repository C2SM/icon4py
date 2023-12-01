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

from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common.dimension import EdgeDim, KDim, V2EDim, VertexDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


class TestMoMathDivrotRotVertexRiDsl(StencilTest):
    PROGRAM = mo_math_divrot_rot_vertex_ri_dsl
    OUTPUTS = ("rot_vec",)

    @staticmethod
    def reference(grid, vec_e: np.array, geofac_rot: np.array, **kwargs) -> dict:
        v2e = grid.connectivities[V2EDim]
        geofac_rot = np.expand_dims(geofac_rot, axis=-1)
        rot_vec = np.sum(
            np.where((v2e != -1)[:, :, np.newaxis], vec_e[v2e] * geofac_rot, 0), axis=1
        )
        return dict(rot_vec=rot_vec)

    @pytest.fixture
    def input_data(self, grid):
        vec_e = random_field(grid, EdgeDim, KDim, dtype=wpfloat)
        geofac_rot = random_field(grid, VertexDim, V2EDim, dtype=wpfloat)
        rot_vec = zero_field(grid, VertexDim, KDim, dtype=vpfloat)

        return dict(
            vec_e=vec_e,
            geofac_rot=geofac_rot,
            rot_vec=rot_vec,
            horizontal_start=int32(0),
            horizontal_end=int32(grid.num_vertices),
            vertical_start=int32(0),
            vertical_end=int32(grid.num_levels),
        )
