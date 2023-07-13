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

from icon4py.model.atmosphere.dycore.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common.dimension import CellDim, KDim

from icon4py.model.common.test_utils.helpers import random_field, zero_field
from icon4py.model.common.test_utils.stencil_test import StencilTest


class TestCalculateDiagnosticsForTurbulence(StencilTest):
    PROGRAM = calculate_diagnostics_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @staticmethod
    def reference(
        mesh, wgtfac_c: np.array, div: np.array, kh_c: np.array, div_ic, hdef_ic
    ) -> tuple[np.array, np.array]:
        kc_offset_1 = np.roll(kh_c, shift=1, axis=1)
        div_offset_1 = np.roll(div, shift=1, axis=1)
        div_ic[:, 1:] = (wgtfac_c * div + (1.0 - wgtfac_c) * div_offset_1)[:, 1:]
        hdef_ic[:, 1:] = ((wgtfac_c * kh_c + (1.0 - wgtfac_c) * kc_offset_1) ** 2)[
            :, 1:
        ]
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfac_c = random_field(mesh, CellDim, KDim)
        div = random_field(mesh, CellDim, KDim)
        kh_c = random_field(mesh, CellDim, KDim)
        div_ic = zero_field(mesh, CellDim, KDim)
        hdef_ic = zero_field(mesh, CellDim, KDim)
        return dict(
            wgtfac_c=wgtfac_c, div=div, kh_c=kh_c, div_ic=div_ic, hdef_ic=hdef_ic
        )
