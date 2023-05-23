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

from icon4py.atm_dyn_iconam.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.common.dimension import CellDim, KDim, V2CDim, VertexDim

from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
    v2c: np.array, p_cell_in: np.array, c_intp: np.array
) -> np.array:
    c_intp = np.expand_dims(c_intp, axis=-1)
    p_vert_out = np.sum(p_cell_in[v2c] * c_intp, axis=1)
    return p_vert_out


def test_mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl():
    mesh = SimpleMesh()

    p_cell_in = random_field(mesh, CellDim, KDim)
    c_intp = random_field(mesh, VertexDim, V2CDim)
    p_vert_out = zero_field(mesh, VertexDim, KDim)

    ref = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
        mesh.v2c, np.asarray(p_cell_in), np.asarray(c_intp)
    )
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl(
        p_cell_in,
        c_intp,
        p_vert_out,
        offset_provider={"V2C": mesh.get_v2c_offset_provider()},
    )

    assert np.allclose(p_vert_out, ref)
