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

from icon4py.model.atmosphere.dycore.apply_diffusion_to_vn import (apply_diffusion_to_vn)
from icon4py.model.common.dimension import EdgeDim, VertexDim, ECVDim, E2C2VDim, KDim

from icon4py.model.common.test_utils.helpers import as_1D_sparse_field, random_field, zero_field, StencilTest


class TestApplyDiffusionToVn(StencilTest):
    PROGRAM = apply_diffusion_to_vn
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        mesh,
        **kwargs,
    ) -> tuple[np.array]:
        vn = 0.
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, mesh):

        horz_idx = zero_field(mesh, EdgeDim, dtype=int32)
        for edge in range(mesh.n_edges):
            horz_idx[edge] = edge

        u_vert = random_field(mesh, VertexDim, KDim)
        v_vert = random_field(mesh, VertexDim, KDim)

        primal_normal_vert_v1 = random_field(mesh, EdgeDim, E2C2VDim)
        primal_normal_vert_v2 = random_field(mesh, EdgeDim, E2C2VDim)

        primal_normal_vert_v1_new = as_1D_sparse_field(primal_normal_vert_v1, ECVDim)
        primal_normal_vert_v2_new = as_1D_sparse_field(primal_normal_vert_v2, ECVDim)

        inv_vert_vert_length = random_field(mesh, EdgeDim)
        inv_primal_edge_length = random_field(mesh, EdgeDim)

        area_edge = random_field(mesh, EdgeDim)
        kh_smag_e = random_field(mesh, EdgeDim, KDim)
        z_nabla2_e = random_field(mesh, EdgeDim, KDim)
        diff_multfac_vn = random_field(mesh, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        nudgecoeff_e = random_field(mesh, EdgeDim)

        limited_area = True
        fac_bdydiff_v = 5.0
        nudgezone_diff = 9.0

        start_2nd_nudge_line_idx_e = 6

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1_new,
            primal_normal_vert_v2=primal_normal_vert_v2_new,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=nudgecoeff_e,
            vn=vn,
            horz_idx=horz_idx,
            nudgezone_diff=nudgezone_diff,
            fac_bdydiff_v=fac_bdydiff_v,
            start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
            limited_area=limited_area,
        )
