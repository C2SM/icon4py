# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    calculate_nabla2_for_z,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def calculate_nabla2_for_z_numpy(
    connectivities: Mapping[gtx.FieldOffset, np.ndarray],
    kh_smag_e: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    theta_v: np.ndarray,
    z_nabla2_e: np.ndarray,
    **kwargs,
) -> np.ndarray:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    theta_v_e2c = theta_v[connectivities[dims.E2C]]
    theta_v_weighted = theta_v_e2c[:, 1] - theta_v_e2c[:, 0]

    z_nabla2_e = kh_smag_e * inv_dual_edge_length * theta_v_weighted

    return z_nabla2_e


class TestCalculateNabla2ForZ(StencilTest):
    PROGRAM = calculate_nabla2_for_z
    OUTPUTS = ("z_nabla2_e",)

    @static_reference
    def reference(
        grid: base.Grid,
        kh_smag_e: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        theta_v: np.ndarray,
        z_nabla2_e: np.ndarray,
        **kwargs,
    ) -> dict:
        connectivities = stencil_tests.connectivities_asnumpy(grid)
        z_nabla2_e = calculate_nabla2_for_z_numpy(
            connectivities, kh_smag_e, inv_dual_edge_length, theta_v, z_nabla2_e, **kwargs
        )
        return dict(z_nabla2_e=z_nabla2_e)

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        kh_smag_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_dual_edge_length = self.data_alloc.random_field(dims.EdgeDim, dtype=wpfloat)
        theta_v = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        z_nabla2_e = self.data_alloc.random_field(dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            kh_smag_e=kh_smag_e,
            inv_dual_edge_length=inv_dual_edge_length,
            theta_v=theta_v,
            z_nabla2_e=z_nabla2_e,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
