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

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_and_nabla4_to_vn import (
    apply_nabla2_and_nabla4_to_vn,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests


def apply_nabla2_and_nabla4_to_vn_numpy(
    area_edge,
    kh_smag_e,
    z_nabla2_e,
    z_nabla4_e2,
    diff_multfac_vn,
    nudgecoeff_e,
    vn,
    nudgezone_diff,
):
    area_edge = np.expand_dims(area_edge, axis=-1)
    diff_multfac_vn = np.expand_dims(diff_multfac_vn, axis=0)
    nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
    vn = vn + area_edge * (
        np.maximum(nudgezone_diff * nudgecoeff_e, kh_smag_e) * z_nabla2_e
        - diff_multfac_vn * z_nabla4_e2 * area_edge
    )
    return vn


class TestApplyNabla2AndNabla4ToVn(stencil_tests.StencilTest):
    PROGRAM = apply_nabla2_and_nabla4_to_vn
    OUTPUTS = ("vn",)

    @stencil_tests.input_data_fixture
    def input_data(self, grid: base.Grid):
        area_edge = self.data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        kh_smag_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        z_nabla2_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        z_nabla4_e2 = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        diff_multfac_vn = self.data_alloc.random_field(dims.KDim, dtype=wpfloat)
        nudgecoeff_e = self.data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        vn = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)
        nudgezone_diff = vpfloat("9.0")

        return dict(
            area_edge=area_edge,
            kh_smag_e=kh_smag_e,
            z_nabla2_e=z_nabla2_e,
            z_nabla4_e2=z_nabla4_e2,
            diff_multfac_vn=diff_multfac_vn,
            nudgecoeff_e=nudgecoeff_e,
            vn=vn,
            nudgezone_diff=nudgezone_diff,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        area_edge: np.ndarray,
        kh_smag_e: np.ndarray,
        z_nabla2_e: np.ndarray,
        z_nabla4_e2: np.ndarray,
        diff_multfac_vn: np.ndarray,
        nudgecoeff_e: np.ndarray,
        vn: np.ndarray,
        nudgezone_diff: np.ndarray,
        **kwargs,
    ) -> dict:
        vn = apply_nabla2_and_nabla4_to_vn_numpy(
            area_edge,
            kh_smag_e,
            z_nabla2_e,
            z_nabla4_e2,
            diff_multfac_vn,
            nudgecoeff_e,
            vn,
            nudgezone_diff,
        )
        return dict(vn=vn)
