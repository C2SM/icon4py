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

from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.common.dimension import EdgeDim, KDim, V2EDim, VertexDim

from .simple_mesh import SimpleMesh
from .utils import random_field, zero_field


def mo_intp_rbf_rbf_vec_interpol_vertex_numpy(
    v2e: np.array, p_e_in: np.array, ptr_coeff_1: np.array, ptr_coeff_2: np.array
) -> tuple[np.array]:
    ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
    p_v_out = np.sum(p_e_in[v2e] * ptr_coeff_1, axis=1)

    ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
    p_u_out = np.sum(p_e_in[v2e] * ptr_coeff_2, axis=1)

    return p_v_out, p_u_out


def test_mo_intp_rbf_rbf_vec_interpol_vertex():
    mesh = SimpleMesh()

    p_e_in = random_field(mesh, EdgeDim, KDim)
    ptr_coeff_1 = random_field(mesh, VertexDim, V2EDim)
    ptr_coeff_2 = random_field(mesh, VertexDim, V2EDim)
    p_v_out = zero_field(mesh, VertexDim, KDim)
    p_u_out = zero_field(mesh, VertexDim, KDim)

    p_v_out_ref, p_u_out_ref = mo_intp_rbf_rbf_vec_interpol_vertex_numpy(
        mesh.v2e, np.asarray(p_e_in), np.asarray(ptr_coeff_1), np.asarray(ptr_coeff_2)
    )
    mo_intp_rbf_rbf_vec_interpol_vertex(
        p_e_in,
        ptr_coeff_1,
        ptr_coeff_2,
        p_v_out,
        p_u_out,
        offset_provider={"V2E": mesh.get_v2e_offset_provider()},
    )

    assert np.allclose(p_v_out, p_v_out_ref)
    assert np.allclose(p_u_out, p_u_out_ref)
