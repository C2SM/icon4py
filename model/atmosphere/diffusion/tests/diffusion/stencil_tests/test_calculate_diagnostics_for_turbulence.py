# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.diffusion.stencils.calculate_diagnostics_for_turbulence import (
    calculate_diagnostics_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import vpfloat
from icon4py.model.testing import stencil_tests


def calculate_diagnostics_for_turbulence_numpy(
    wgtfac_c: np.ndarray, div: np.ndarray, kh_c: np.ndarray, div_ic, hdef_ic
) -> tuple[np.ndarray, np.ndarray]:
    nlev = div.shape[1]
    w = wgtfac_c[:, 1:nlev]
    div_ic[:, 1:nlev] = w * div[:, 1:nlev] + (1.0 - w) * div[:, 0 : nlev - 1]
    hdef_ic[:, 1:nlev] = (w * kh_c[:, 1:nlev] + (1.0 - w) * kh_c[:, 0 : nlev - 1]) ** 2
    return div_ic, hdef_ic


class TestCalculateDiagnosticsForTurbulence(stencil_tests.StencilTest):
    PROGRAM = calculate_diagnostics_for_turbulence
    OUTPUTS = ("div_ic", "hdef_ic")

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        wgtfac_c: np.ndarray,
        div: np.ndarray,
        kh_c: np.ndarray,
        div_ic: np.ndarray,
        hdef_ic: np.ndarray,
        **kwargs: object,
    ) -> dict:
        div_ic, hdef_ic = calculate_diagnostics_for_turbulence_numpy(
            wgtfac_c, div, kh_c, div_ic, hdef_ic
        )
        return dict(div_ic=div_ic, hdef_ic=hdef_ic)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        wgtfac_c = data_alloc.random_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)
        div = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        kh_c = data_alloc.random_field(dims.CellDim, dims.KDim, dtype=vpfloat)
        div_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)
        hdef_ic = data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=vpfloat)
        return dict(wgtfac_c=wgtfac_c, div=div, kh_c=kh_c, div_ic=div_ic, hdef_ic=hdef_ic)
