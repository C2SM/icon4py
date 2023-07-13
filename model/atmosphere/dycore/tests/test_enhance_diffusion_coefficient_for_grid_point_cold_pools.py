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

from icon4py.model.atmosphere.dycore.enhance_diffusion_coefficient_for_grid_point_cold_pools import (
    enhance_diffusion_coefficient_for_grid_point_cold_pools,
)
from icon4py.model.common.dimension import CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import random_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestEnhanceDiffusionCoefficientForGridPointColdPools(StencilTest):
    PROGRAM = enhance_diffusion_coefficient_for_grid_point_cold_pools
    OUTPUTS = ("kh_smag_e",)

    @staticmethod
    def reference(
        mesh,
        kh_smag_e: np.array,
        enh_diffu_3d: np.array,
    ) -> np.array:
        kh_smag_e = np.maximum(kh_smag_e, np.max(enh_diffu_3d[mesh.e2c], axis=1))
        return dict(kh_smag_e=kh_smag_e)

    @pytest.fixture
    def input_data(self, mesh):
        kh_smag_e = random_field(mesh, EdgeDim, KDim)
        enh_diffu_3d = random_field(mesh, CellDim, KDim)

        return dict(
            kh_smag_e=kh_smag_e,
            enh_diffu_3d=enh_diffu_3d,
        )
