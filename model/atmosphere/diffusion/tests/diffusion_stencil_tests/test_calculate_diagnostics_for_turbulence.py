# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.settings import xp
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field
from icon4py.model.common.type_alias import vpfloat


def calculate_diagnostics_for_turbulence_numpy(
    wgtfac_c: xp.array, div: xp.array, kh_c: xp.array, div_ic, hdef_ic
) -> tuple[xp.array, xp.array]:
    div = xp.asarray(div)
    wgtfac_c = xp.asarray(wgtfac_c)
    kc_offset_1 = xp.roll(xp.asarray(kh_c), shift=1, axis=1)
    div_offset_1 = xp.roll(div, shift=1, axis=1)
    div_ic[:, 1:] = (wgtfac_c * div + (1.0 - wgtfac_c) * div_offset_1)[:, 1:]
    hdef_ic[:, 1:] = ((wgtfac_c * kh_c + (1.0 - wgtfac_c) * kc_offset_1) ** 2)[:, 1:]
    return div_ic, hdef_ic


class TestCalculateDiagnosticsForTurbulence(StencilTest):
    PROGRAM = calculate_diagnostics_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @staticmethod
    def reference(grid, wgtfac_c: xp.array, div: xp.array, kh_c: xp.array, div_ic, hdef_ic) -> dict:
        div_ic, hdef_ic = calculate_diagnostics_for_turbulence_numpy(
            wgtfac_c, div, kh_c, div_ic, hdef_ic
        )
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @pytest.fixture
    def input_data(self, grid):
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        div = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        kh_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        div_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        hdef_ic = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        return dict(wgtfac_c=wgtfac_c, div=div, kh_c=kh_c, div_ic=div_ic, hdef_ic=hdef_ic)
