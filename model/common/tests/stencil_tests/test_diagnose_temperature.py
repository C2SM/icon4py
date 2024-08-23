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

from icon4py.model.common import constants as phy_const, dimension as dims, type_alias as ta
from icon4py.model.common.diagnostic_calculations.stencils.diagnose_temperature import (
    diagnose_virtual_temperature_and_temperature,
)
from icon4py.model.common.test_utils import helpers


class TestDiagnoseTemperature(helpers.StencilTest):
    PROGRAM = diagnose_virtual_temperature_and_temperature
    OUTPUTS = ("virtual_temperature", "temperature")

    @staticmethod
    def reference(
        grid,
        qv: np.array,
        qc: np.array,
        qi: np.array,
        qr: np.array,
        qs: np.array,
        qg: np.array,
        theta_v: np.array,
        exner: np.array,
        **kwargs,
    ) -> dict:
        qsum = qc + qi + qr + qs + qg
        virtual_temperature = theta_v * exner
        temperature = virtual_temperature / (1.0 + phy_const.RV_O_RD_MINUS_1 * qv - qsum)
        return dict(
            virtual_temperature=virtual_temperature,
            temperature=temperature,
        )

    @pytest.fixture
    def input_data(self, grid):
        theta_v = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-6, high=1.0, dtype=ta.wpfloat
        )
        exner = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=1.0e-6, high=1.0, dtype=ta.wpfloat
        )
        qv = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qc = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qi = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qr = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qs = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        qg = helpers.random_field(
            grid, dims.CellDim, dims.KDim, low=0.0, high=1.0, dtype=ta.wpfloat
        )
        virtual_temperature = helpers.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        temperature = helpers.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
            qg=qg,
            theta_v=theta_v,
            exner=exner,
            virtual_temperature=virtual_temperature,
            temperature=temperature,
            rv_o_rd_minus1=phy_const.RV_O_RD_MINUS_1,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
