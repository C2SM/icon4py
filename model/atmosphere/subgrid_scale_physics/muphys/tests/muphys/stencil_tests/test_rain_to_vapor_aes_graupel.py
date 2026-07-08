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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import (
    rain_to_vapor_aes_graupel,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestRainToVaporAesGraupel(StencilTest):
    PROGRAM = rain_to_vapor_aes_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        rho: np.ndarray,
        qc: np.ndarray,
        qr: np.ndarray,
        dvsw: np.ndarray,
        dt: np.ndarray,
        **kwargs,
    ) -> dict:
        # mirrors icon-nwp mo_aes_graupel.f90 rain_to_vapor
        a_ev = [-5.532194e00, 2.432848e-01, -4.145391e-02, -1.798439e-03, -1.405764e-05]
        tc = t - 273.15
        evap_max = (0.61 + tc * (-0.0163 + 1.111e-4 * tc)) * (-dvsw) / dt
        x = np.log(np.clip(qr * rho, 3.26216e-08, 6.97604e-03))
        evap = -np.exp(a_ev[0] + x * (a_ev[1] + x * (a_ev[2] + x * (a_ev[3] + x * a_ev[4])))) * dvsw
        rate = np.where((qr > 1.0e-15) & (dvsw + qc <= 0.0), np.minimum(evap, evap_max), 0.0)
        return dict(conversion_rate=rate)

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 258.542, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 0.956089, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=data_alloc.constant_field(grid, 3.01332e-11, dims.CellDim, dims.KDim, dtype=wpfloat),
            dvsw=data_alloc.constant_field(grid, -1.0e-10, dims.CellDim, dims.KDim, dtype=wpfloat),
            dt=30.0,
            conversion_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
