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

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.common.dimension import E2C2VDim, ECVDim, EdgeDim, KDim, VertexDim
from icon4py.model.common.test_utils.helpers import (
    StencilTest,
    as_1D_sparse_field,
    random_field,
    zero_field,
)
from icon4py.model.atmosphere.dycore.state_utils.utils import indices_field

from .test_apply_nabla2_and_nabla4_global_to_vn import apply_nabla2_and_nabla4_global_to_vn_numpy
from .test_apply_nabla2_and_nabla4_to_vn import apply_nabla2_and_nabla4_to_vn_numpy
from .test_apply_nabla2_to_vn_in_lateral_boundary import (
    apply_nabla2_to_vn_in_lateral_boundary_numpy,
)
from .test_calculate_nabla4 import calculate_nabla4_numpy


class TestApplyDiffusionToVn(StencilTest):
    PROGRAM = apply_diffusion_to_vn
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        grid,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        area_edge,
        kh_smag_e,
        diff_multfac_vn,
        nudgecoeff_e,
        vn,
        edge,
        nudgezone_diff,
        fac_bdydiff_v,
        start_2nd_nudge_line_idx_e,
        limited_area,
        **kwargs,
    ):

        z_nabla4_e2 = calculate_nabla4_numpy(
            grid,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        )

        condition = start_2nd_nudge_line_idx_e <= edge[:, np.newaxis]

        if limited_area:
            vn = np.where(
                condition,
                apply_nabla2_and_nabla4_to_vn_numpy(
                    grid,
                    area_edge,
                    kh_smag_e,
                    z_nabla2_e,
                    z_nabla4_e2,
                    diff_multfac_vn,
                    nudgecoeff_e,
                    vn,
                    nudgezone_diff,
                ),
                apply_nabla2_to_vn_in_lateral_boundary_numpy(
                    grid, z_nabla2_e, area_edge, vn, fac_bdydiff_v
                ),
            )
        else:
            vn = np.where(
                condition,
                apply_nabla2_and_nabla4_global_to_vn_numpy(
                    grid, area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn
                ),
                vn,
            )

        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        edge = indices_field(EdgeDim, grid, is_halfdim=False, dtype=int32)

        u_vert = random_field(grid, VertexDim, KDim)
        v_vert = random_field(grid, VertexDim, KDim)

        primal_normal_vert_v1 = random_field(grid, EdgeDim, E2C2VDim)
        primal_normal_vert_v2 = random_field(grid, EdgeDim, E2C2VDim)

        primal_normal_vert_v1_new = as_1D_sparse_field(primal_normal_vert_v1, ECVDim)
        primal_normal_vert_v2_new = as_1D_sparse_field(primal_normal_vert_v2, ECVDim)

        inv_vert_vert_length = random_field(grid, EdgeDim)
        inv_primal_edge_length = random_field(grid, EdgeDim)

        area_edge = random_field(grid, EdgeDim)
        kh_smag_e = random_field(grid, EdgeDim, KDim)
        z_nabla2_e = random_field(grid, EdgeDim, KDim)
        diff_multfac_vn = random_field(grid, KDim)
        vn = random_field(grid, EdgeDim, KDim)
        nudgecoeff_e = random_field(grid, EdgeDim)

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
            edge=edge,
            nudgezone_diff=nudgezone_diff,
            fac_bdydiff_v=fac_bdydiff_v,
            start_2nd_nudge_line_idx_e=start_2nd_nudge_line_idx_e,
            limited_area=limited_area,
            horizontal_start=0,
            horizontal_end=grid.num_edges,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
