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

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common.dimension import CellDim, KDim, KHalfDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


def calculate_diagnostics_for_turbulence_numpy(
    wgtfac_c: np.array, div: np.array, kh_c: np.array, div_ic, hdef_ic
) -> tuple[np.array, np.array]:
    kc_offset_1 = np.roll(kh_c, shift=1, axis=1)
    div_offset_1 = np.roll(div, shift=1, axis=1)
    div_ic = (wgtfac_c[:, 1:] * div + (1.0 - wgtfac_c[:, 1:]) * div_offset_1)
    hdef_ic = ((wgtfac_c[:, 1:] * kh_c + (1.0 - wgtfac_c[:, 1:]) * kc_offset_1) ** 2)
    return div_ic, hdef_ic


class TestCalculateDiagnosticsForTurbulence(StencilTest):
    PROGRAM = calculate_diagnostics_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @staticmethod
    def reference(grid, wgtfac_c: np.array, div: np.array, kh_c: np.array, div_ic, hdef_ic,
                    horizontal_start = 0,
                    horizontal_end=0,
                    vertical_start = 0,
                    vertical_end = 0,

    ) -> dict:
        div_ic, hdef_ic = calculate_diagnostics_for_turbulence_numpy(
            wgtfac_c, div, kh_c, div_ic, hdef_ic
        )
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_c = random_field(grid, CellDim, KHalfDim, dtype=vpfloat)
        div = random_field(grid, CellDim, KDim, dtype=vpfloat)
        kh_c = random_field(grid, CellDim, KDim, dtype=vpfloat)
        div_ic = zero_field(grid, CellDim, KHalfDim, dtype=vpfloat)
        hdef_ic = zero_field(grid, CellDim, KHalfDim, dtype=vpfloat)
        return dict(wgtfac_c=wgtfac_c,
                    div=div,
                    kh_c=kh_c,
                    div_ic=div_ic,
                    hdef_ic=hdef_ic,
                    horizontal_start=0,
                    horizontal_end=grid.num_cells,
                    vertical_start=0,
                    vertical_end=grid.num_levels,
                    )
