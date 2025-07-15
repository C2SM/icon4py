# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.helpers import StencilTest


class TestIceToSnow(StencilTest):
    PROGRAM = ice_to_snow
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(
        grid, qi: np.ndarray, ns: np.ndarray, lam: np.ndarray, sticking_eff: np.ndarray, **kwargs
    ) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 3.3262745200740486e-11))

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            qi=data_alloc.constant_field(grid, 6.43223e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=data_alloc.constant_field(grid, 1.93157e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=data_alloc.constant_field(grid, 10576.8, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff=data_alloc.constant_field(
                grid, 0.511825, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            conversion_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
