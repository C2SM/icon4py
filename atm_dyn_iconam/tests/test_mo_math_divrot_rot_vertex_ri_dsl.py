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

from icon4py.atm_dyn_iconam.mo_math_divrot_rot_vertex_ri_dsl import (
    mo_math_divrot_rot_vertex_ri_dsl,
)
from icon4py.common.dimension import EdgeDim, KDim, V2EDim, VertexDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.simple_mesh import SimpleMesh


def mo_math_divrot_rot_vertex_ri_dsl_numpy(
    v2e: np.array, vec_e: np.array, geofac_rot: np.array
) -> np.array:
    geofac_rot = np.expand_dims(geofac_rot, axis=-1)
    rot_vec = np.sum(vec_e[v2e] * geofac_rot, axis=1)
    return rot_vec


def test_mo_math_divrot_rot_vertex_ri_dsl_numpy():
    mesh = SimpleMesh()

    vec_e = random_field(mesh, EdgeDim, KDim)
    geofac_rot = random_field(mesh, VertexDim, V2EDim)
    rot_vec = zero_field(mesh, VertexDim, KDim)

    ref = mo_math_divrot_rot_vertex_ri_dsl_numpy(
        mesh.v2e,
        np.asarray(vec_e),
        np.asarray(geofac_rot),
    )

    mo_math_divrot_rot_vertex_ri_dsl(
        vec_e,
        geofac_rot,
        rot_vec,
        offset_provider={"V2E": mesh.get_v2e_offset_provider(), "V2EDim": V2EDim},
    )

    assert np.allclose(rot_vec, ref)
