# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


def add_vertical_wind_derivative_to_divergence_damping_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    horizontal_mask_for_3d_divdamp: np.ndarray,
    scaling_factor_for_3d_divdamp: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    z_dwdz_dd: np.ndarray,
    z_graddiv_vn: np.ndarray,
) -> np.ndarray:
    scaling_factor_for_3d_divdamp = np.expand_dims(scaling_factor_for_3d_divdamp, axis=0)
    horizontal_mask_for_3d_divdamp = np.expand_dims(horizontal_mask_for_3d_divdamp, axis=-1)
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)

    e2c = connectivities[dims.E2CDim]
    z_dwdz_dd_e2c = z_dwdz_dd[e2c]
    z_dwdz_dd_weighted = z_dwdz_dd_e2c[:, 1] - z_dwdz_dd_e2c[:, 0]

    z_graddiv_vn = z_graddiv_vn + (
        horizontal_mask_for_3d_divdamp * scaling_factor_for_3d_divdamp * inv_dual_edge_length * z_dwdz_dd_weighted
    )
    return z_graddiv_vn


@pytest.mark.skip_value_error
class TestAddVerticalWindDerivativeToDivergenceDamping(stencil_tests.StencilTest):
    PROGRAM = add_vertical_wind_derivative_to_divergence_damping
    OUTPUTS = ("z_graddiv_vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        horizontal_mask_for_3d_divdamp: np.ndarray,
        scaling_factor_for_3d_divdamp: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        z_dwdz_dd: np.ndarray,
        z_graddiv_vn: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        z_graddiv_vn = add_vertical_wind_derivative_to_divergence_damping_numpy(
            connectivities,
            horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp,
            inv_dual_edge_length,
            z_dwdz_dd,
            z_graddiv_vn,
        )
        return dict(z_graddiv_vn=z_graddiv_vn)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        horizontal_mask_for_3d_divdamp = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        scaling_factor_for_3d_divdamp = data_alloc.random_field(grid, dims.KDim, dtype=ta.wpfloat)
        inv_dual_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        z_dwdz_dd = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_graddiv_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
            scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
            inv_dual_edge_length=inv_dual_edge_length,
            z_dwdz_dd=z_dwdz_dd,
            z_graddiv_vn=z_graddiv_vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
