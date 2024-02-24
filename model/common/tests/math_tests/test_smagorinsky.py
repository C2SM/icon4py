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

from icon4py.model.common.dimension import KDim
from icon4py.model.common.math.smagorinsky import en_smag_fac_for_zero_nshift
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.test_utils.reference_funcs import enhanced_smagorinski_factor_numpy


class TestEnhancedSmagorinskiFactor(StencilTest):
    PROGRAM = en_smag_fac_for_zero_nshift
    OUTPUTS = ("enh_smag_fac",)

    @staticmethod
    def reference(
        grid,
        vect_a: np.ndarray,
        hdiff_smag_fac: float,
        hdiff_smag_fac2: float,
        hdiff_smag_fac3: float,
        hdiff_smag_fac4: float,
        hdiff_smag_z: float,
        hdiff_smag_z2: float,
        hdiff_smag_z3: float,
        hdiff_smag_z4: float,
        **kwargs,
    ):
        fac = (hdiff_smag_fac, hdiff_smag_fac2, hdiff_smag_fac3, hdiff_smag_fac4)
        z = (hdiff_smag_z, hdiff_smag_z2, hdiff_smag_z3, hdiff_smag_z4)
        enh_smag_fac = enhanced_smagorinski_factor_numpy(fac, z, vect_a)
        return dict(enh_smag_fac=enh_smag_fac)

    @pytest.fixture
    def input_data(self, grid):
        enh_smag_fac = zero_field(grid, KDim)
        a_vec = random_field(grid, KDim, low=1.0, high=10.0, extend={KDim: 1})
        fac = (0.67, 0.5, 1.3, 0.8)
        z = (0.1, 0.2, 0.3, 0.4)

        return dict(
            enh_smag_fac=enh_smag_fac,
            vect_a=a_vec,
            hdiff_smag_fac=fac[0],
            hdiff_smag_fac2=fac[1],
            hdiff_smag_fac3=fac[2],
            hdiff_smag_fac4=fac[3],
            hdiff_smag_z=z[0],
            hdiff_smag_z2=z[1],
            hdiff_smag_z3=z[2],
            hdiff_smag_z4=z[3],
        )
