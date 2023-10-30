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

from icon4py.model.atmosphere.dycore.calculate_diagnostic_quantities_for_turbulence import (
    calculate_diagnostic_quantities_for_turbulence,
)
from icon4py.model.common.dimension import C2EDim, CellDim, EdgeDim, KDim
from icon4py.model.common.test_utils.helpers import StencilTest, random_field, zero_field


class TestCalculateDiagnosticQuantitiesForTurbulence(StencilTest):
    PROGRAM = calculate_diagnostic_quantities_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @staticmethod
    def reference(
        mesh,
        wgtfac_c: np.array,
        kh_smag_ec: np.array,
        vn: np.array,
        e_bln_c_s: np.array,
        geofac_div: np.array,
        diff_multfac_smag: np.array,
        div_ic,
        hdef_ic,
    ) -> tuple[np.array, np.array]:
        div_ic = 0.0
        hdef_ic = 0.0
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @pytest.fixture
    def input_data(self, mesh):
        wgtfac_c = random_field(mesh, CellDim, KDim)
        vn = random_field(mesh, EdgeDim, KDim)
        geofac_div = random_field(mesh, CellDim, C2EDim)
        kh_smag_ec = random_field(mesh, EdgeDim, KDim)
        e_bln_c_s = random_field(mesh, CellDim, C2EDim)
        diff_multfac_smag = random_field(mesh, KDim)

        div_ic = zero_field(mesh, CellDim, KDim)
        hdef_ic = zero_field(mesh, CellDim, KDim)
        return dict(
            wgtfac_c=wgtfac_c,
            vn=vn,
            geofac_div=geofac_div,
            kh_smag_ec=kh_smag_ec,
            e_bln_c_s=e_bln_c_s,
            diff_multfac_smag=diff_multfac_smag,
            div_ic=div_ic,
            hdef_ic=hdef_ic,
        )
