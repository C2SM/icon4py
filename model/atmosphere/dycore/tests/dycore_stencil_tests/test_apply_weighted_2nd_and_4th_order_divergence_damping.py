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

from icon4py.model.atmosphere.dycore.stencils.apply_weighted_2nd_and_4th_order_divergence_damping import (
    apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


class TestApplyWeighted2ndAnd4thOrderDivergenceDamping(StencilTest):
    PROGRAM = apply_weighted_2nd_and_4th_order_divergence_damping
    OUTPUTS = ("vn",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        scal_divdamp: np.ndarray,
        bdy_divdamp: np.ndarray,
        nudgecoeff_e: np.ndarray,
        z_graddiv2_vn: np.ndarray,
        vn: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        nudgecoeff_e = np.expand_dims(nudgecoeff_e, axis=-1)
        vn = vn + (scal_divdamp + bdy_divdamp * nudgecoeff_e) * z_graddiv2_vn
        return dict(vn=vn)

    @pytest.fixture
    def input_data(self, grid: base.BaseGrid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        scal_divdamp = random_field(grid, dims.KDim, dtype=wpfloat)
        bdy_divdamp = random_field(grid, dims.KDim, dtype=wpfloat)
        nudgecoeff_e = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_graddiv2_vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        vn = random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)

        return dict(
            scal_divdamp=scal_divdamp,
            bdy_divdamp=bdy_divdamp,
            nudgecoeff_e=nudgecoeff_e,
            z_graddiv2_vn=z_graddiv2_vn,
            vn=vn,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
