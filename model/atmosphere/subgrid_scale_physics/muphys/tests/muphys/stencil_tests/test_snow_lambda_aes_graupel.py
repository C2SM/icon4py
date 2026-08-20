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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.properties import (
    snow_lambda_aes_graupel,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestSnowLambdaAesGraupel(stencil_tests.StencilTest):
    PROGRAM = snow_lambda_aes_graupel
    OUTPUTS = ("riming_snow_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rho_s: np.ndarray,
        ns: np.ndarray,
        **kwargs,
    ) -> dict:
        # mirrors ICON mo_aes_graupel.f90 snow_lambda
        lam = np.where(rho_s > 1.0e-15, (2.0 * 0.069 * ns / rho_s) ** (1.0 / 3.0), 1.0e10)
        return dict(riming_snow_rate=lam)

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            rho_s=data_alloc.constant_field(
                1.12204 * 7.47365e-06, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            ns=data_alloc.constant_field(1.76669e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
