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
    cloud_to_rain_aes_graupel,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestCloudToRainAesGraupel(StencilTest):
    PROGRAM = cloud_to_rain_aes_graupel
    OUTPUTS = ("conversion_rate",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        t: np.ndarray,
        rho: np.ndarray,
        qc: np.ndarray,
        qr: np.ndarray,
        nc: np.ndarray,
        **kwargs,
    ) -> dict:
        # independent numpy reference mirroring icon-mpim mo_aes_graupel.f90 cloud_to_rain;
        # the magic constants are hardcoded PARAMETERs there in the same way
        au_kernel = 9.44e9 / (20.0 * 2.6e-10) * 4.0 * 6.0 / 9.0
        a_ac = [-2.155543e00, -1.148491e00, -1.882563e-02, 2.941391e-03, 5.575598e-05]
        x = np.log(np.clip(rho * qr, 3.26216e-08, 6.97604e-03))
        ac_kernel = a_ac[0] + x * (a_ac[1] + x * (a_ac[2] + x * (a_ac[3] + x * a_ac[4])))
        tau = np.maximum(1.0e-30, np.minimum(1.0 - qc / (qc + qr), 0.9))
        phi = tau**0.68
        phi = 6.0e2 * phi * (1.0 - phi) ** 3
        xau = au_kernel * (qc * qc / nc) ** 2 * (1.0 + phi / (1.0 - tau) ** 2)
        xac = ac_kernel * qc * qr * (tau / (tau + 5.0e-5)) ** 4
        rate = np.where((qc > 1.0e-6) & (t > 236.15), xau + xac, 0.0)
        return dict(conversion_rate=rate)

    @pytest.fixture
    def input_data(self, grid):
        return dict(
            t=data_alloc.constant_field(grid, 267.25, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(grid, 0.956089, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(grid, 5.52921e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=data_alloc.constant_field(grid, 2.01511e-12, dims.CellDim, dims.KDim, dtype=wpfloat),
            nc=100.0,
            conversion_rate=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
        )
