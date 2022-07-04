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

from icon4py.atm_dyn_iconam.mo_intp_rbf_rbf_vec_interpol_vertex_dom_a import (
    mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_u,
    mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_v,
)
from icon4py.common.dimension import EdgeDim, KDim, V2EDim, VertexDim
from icon4py.testutils.simple_mesh import SimpleMesh
from icon4py.testutils.utils import random_field, zero_field


def mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_v_numpy(
    v2e: np.array,
    p_e_in: np.array,
    ptr_coeff_1: np.array,
) -> np.array:
    ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
    p_v_out = np.sum(p_e_in[v2e] * ptr_coeff_1, axis=1)
    return p_v_out


def mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_u_numpy(
    v2e: np.array,
    p_e_in: np.array,
    ptr_coeff_2: np.array,
) -> np.array:
    ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
    p_u_out = np.sum(p_e_in[v2e] * ptr_coeff_2, axis=1)
    return p_u_out


def test_mo_intp_rbf_rbf_vec_interpol_vertex_dom_a():
    mesh = SimpleMesh()

    p_e_in = random_field(mesh, EdgeDim, KDim)
    ptr_coeff = random_field(mesh, VertexDim, V2EDim)
    out = zero_field(mesh, VertexDim, KDim)

    stencil_funcs = {
        mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_u_numpy: mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_u,
        mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_v_numpy: mo_intp_rbf_rbf_vec_interpol_vertex_dom_a_p_v,
    }

    for ref_func, func in stencil_funcs.items():
        ref = ref_func(mesh.v2e, np.asarray(p_e_in), np.asarray(ptr_coeff))
        func(
            p_e_in,
            ptr_coeff,
            out,
            offset_provider={"V2E": mesh.get_v2e_offset_provider()},
        )

    assert np.allclose(out, ref)
