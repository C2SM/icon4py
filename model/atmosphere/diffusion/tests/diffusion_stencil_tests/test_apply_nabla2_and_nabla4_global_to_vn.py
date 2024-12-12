# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_global_to_vn import (
    apply_nabla2_and_nabla4_global_to_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.test_utils.helpers import StencilTest, random_field
from icon4py.model.common.type_alias import vpfloat, wpfloat


def apply_nabla2_and_nabla4_global_to_vn_numpy(
    mesh, area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn
):
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    vn = vn + area_edge * (kh_smag_e * z_nabla2_e - diff_multfac_vn * z_nabla4_e2 * area_edge)
    return vn


class TestApplyNabla2AndNabla4GlobalToVn(StencilTest):
    PROGRAM = apply_nabla2_and_nabla4_global_to_vn
    OUTPUTS = ("vn",)

    @pytest.fixture
    def input_data(self, grid):
        area_edge = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        kh_smag_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_nabla2_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_nabla4_e2 = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        diff_multfac_vn = random_field(grid, dims.KDim, dtype=wpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            z_nabla2_e=z_nabla2_e,
            z_nabla4_e2=z_nabla4_e2,
            diff_multfac_vn=diff_multfac_vn,
            vn=vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @staticmethod
    def reference(
        grid, area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn, **kwargs
    ):
        vn = apply_nabla2_and_nabla4_global_to_vn_numpy(
            grid, area_edge, kh_smag_e, z_nabla2_e, z_nabla4_e2, diff_multfac_vn, vn
        )
        return dict(
            vn=vn,
        )
