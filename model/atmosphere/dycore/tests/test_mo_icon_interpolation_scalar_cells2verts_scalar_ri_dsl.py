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

from icon4py.model.atmosphere.dycore.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common.dimension import CellDim, KDim, V2CDim, VertexDim
from icon4py.model.common.test_utils.helpers import random_field, zero_field, StencilTest



class TestMoIconInterpolationScalarCells2vertsScalarRiDsl(StencilTest):
    PROGRAM = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl
    OUTPUTS = ("p_vert_out",)

    @staticmethod
    def reference(mesh, p_cell_in: np.array, c_intp: np.array, **kwargs) -> np.array:
        c_intp = np.expand_dims(c_intp, axis=-1)
        p_vert_out = np.sum(p_cell_in[mesh.v2c] * c_intp, axis=1)
        return dict(p_vert_out=p_vert_out)

    @pytest.fixture
    def input_data(self, mesh):
        p_cell_in = random_field(mesh, CellDim, KDim)
        c_intp = random_field(mesh, VertexDim, V2CDim)
        p_vert_out = zero_field(mesh, VertexDim, KDim)

        return dict(
            p_cell_in=p_cell_in,
            c_intp=c_intp,
            p_vert_out=p_vert_out,
        )
