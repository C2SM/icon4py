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
    snow_number_aes_graupel,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestSnowNumberAesGraupel(StencilTest):
    PROGRAM = snow_number_aes_graupel
    OUTPUTS = ("number",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        rho_s: np.ndarray,
        **kwargs,
    ) -> dict:
        # mirrors icon-nwp mo_aes_graupel.f90 snow_number
        n0s1 = 13.5 * 5.65e05
        tc = np.clip(t, 233.15, 273.15) - 273.15
        alf = 10.0 ** (-1.65 + tc * (5.45e-2 + tc * 3.27e-4))
        bet = 1.42 + tc * (1.19e-2 + tc * 9.60e-5)
        n0s = 13.5 * (np.maximum(rho_s, 2.0e-7) / 0.069) ** (4.0 - 3.0 * bet) / (alf**3)
        y = np.exp(-0.107 * tc)
        n0smn = np.maximum(0.5 * n0s1 * y, 1.0e6)
        n0smx = np.minimum(1.0e2 * n0s1 * y, 1.0e9)
        number = np.where(rho_s > 1.0e-15, np.minimum(n0smx, np.maximum(n0smn, n0s)), 8.00e5)
        return dict(number=number)

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 276.302, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho_s=data_alloc.constant_field(
                grid, 1.17797 * 8.28451e-4, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            number=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
