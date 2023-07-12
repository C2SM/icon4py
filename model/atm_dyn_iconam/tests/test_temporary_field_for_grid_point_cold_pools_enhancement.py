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

from icon4py.model.atm_dyn_iconam.temporary_field_for_grid_point_cold_pools_enhancement import (
    temporary_field_for_grid_point_cold_pools_enhancement,
)
from icon4py.model.common.dimension import CellDim, KDim

from .test_utils.helpers import random_field, zero_field
from .test_utils.stencil_test import StencilTest


class TestTemporaryFieldForGridPointColdPoolsEnhancement(StencilTest):
    PROGRAM = temporary_field_for_grid_point_cold_pools_enhancement
    OUTPUTS = ("enh_diffu_3d",)

    @staticmethod
    def reference(
        mesh, theta_v: np.array, theta_ref_mc: np.array, thresh_tdiff, **kwargs
    ) -> np.array:
        tdiff = theta_v - np.sum(theta_v[mesh.c2e2c], axis=1) / 3
        trefdiff = theta_ref_mc - np.sum(theta_ref_mc[mesh.c2e2c], axis=1) / 3

        enh_diffu_3d = np.where(
            ((tdiff - trefdiff) < thresh_tdiff) & (trefdiff < 0),
            (thresh_tdiff - tdiff + trefdiff) * 5e-4,
            -1.7976931348623157e308,
        )

        return dict(enh_diffu_3d=enh_diffu_3d)

    @pytest.fixture
    def input_data(self, mesh):
        theta_v = random_field(mesh, CellDim, KDim)
        theta_ref_mc = random_field(mesh, CellDim, KDim)
        enh_diffu_3d = zero_field(mesh, CellDim, KDim)
        thresh_tdiff = 5.0

        return dict(
            theta_v=theta_v,
            theta_ref_mc=theta_ref_mc,
            enh_diffu_3d=enh_diffu_3d,
            thresh_tdiff=thresh_tdiff,
        )
