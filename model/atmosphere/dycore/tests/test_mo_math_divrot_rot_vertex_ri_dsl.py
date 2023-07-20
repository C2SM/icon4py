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

from icon4py.model.atmosphere.dycore.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.model.common.dimension import EdgeDim, KDim, V2EDim, VertexDim
from icon4py.model.common.test_utils.helpers import random_field, zero_field, StencilTest



class TestMoMathDivrotRotVertexRiDsl(StencilTest):
    PROGRAM = mo_math_divrot_rot_vertex_ri_dsl
    OUTPUTS = ("rot_vec",)

    @staticmethod
    def reference(mesh, vec_e: np.array, geofac_rot: np.array, **kwargs) -> np.array:
        geofac_rot = np.expand_dims(geofac_rot, axis=-1)
        rot_vec = np.sum(vec_e[mesh.v2e] * geofac_rot, axis=1)
        return dict(rot_vec=rot_vec)

    @pytest.fixture
    def input_data(self, mesh):
        vec_e = random_field(mesh, EdgeDim, KDim)
        geofac_rot = random_field(mesh, VertexDim, V2EDim)
        rot_vec = zero_field(mesh, VertexDim, KDim)

        return dict(
            vec_e=vec_e,
            geofac_rot=geofac_rot,
            rot_vec=rot_vec,
        )
