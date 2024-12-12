# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.apply_diffusion_to_vn import apply_diffusion_to_vn
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.test_utils.helpers import StencilTest, as_1D_sparse_field, random_field
from icon4py.model.common.utils import gt4py_field_allocation as field_alloc

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
        vn_cp = vn.copy()
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

        # restriction of execution domain
        vn[0 : kwargs["horizontal_start"], :] = vn_cp[0 : kwargs["horizontal_start"], :]
        vn[kwargs["horizontal_end"] :, :] = vn_cp[kwargs["horizontal_end"] :, :]

        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid):
        edge = field_alloc.allocate_indices(dims.EdgeDim, grid=grid, is_halfdim=False)

        u_vert = random_field(grid, dims.VertexDim, dims.KDim)
        v_vert = random_field(grid, dims.VertexDim, dims.KDim)

        primal_normal_vert_v1 = random_field(grid, dims.EdgeDim, dims.E2C2VDim)
        primal_normal_vert_v2 = random_field(grid, dims.EdgeDim, dims.E2C2VDim)

        primal_normal_vert_v1_new = as_1D_sparse_field(primal_normal_vert_v1, dims.ECVDim)
        primal_normal_vert_v2_new = as_1D_sparse_field(primal_normal_vert_v2, dims.ECVDim)

        inv_vert_vert_length = random_field(grid, dims.EdgeDim)
        inv_primal_edge_length = random_field(grid, dims.EdgeDim)

        area_edge = random_field(grid, dims.EdgeDim)
        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim)
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim)
        diff_multfac_vn = random_field(grid, dims.KDim)
        vn = random_field(grid, dims.EdgeDim, dims.KDim)
        nudgecoeff_e = random_field(grid, dims.EdgeDim)

        limited_area = grid.limited_area if hasattr(grid, "limited_area") else True
        fac_bdydiff_v = 5.0
        nudgezone_diff = 9.0

        start_2nd_nudge_line_idx_e = 6

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = (
            grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5))
            if hasattr(grid, "start_index")
            else 0
        )
        horizontal_end = (
            grid.end_index(edge_domain(h_grid.Zone.LOCAL))
            if hasattr(grid, "end_index")
            else grid.num_edges
        )

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
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
