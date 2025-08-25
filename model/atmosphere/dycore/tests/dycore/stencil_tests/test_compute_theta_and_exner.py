# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_theta_and_exner import compute_theta_and_exner
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


class TestComputeThetaAndExner(StencilTest):
    PROGRAM = compute_theta_and_exner
    OUTPUTS = ("theta_v", "exner")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        bdy_halo_c: np.ndarray,
        rho: np.ndarray,
        theta_v: np.ndarray,
        exner: np.ndarray,
        rd_o_cvd: ta.wpfloat,
        rd_o_p0ref: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        bdy_halo_c = np.expand_dims(bdy_halo_c, axis=-1)

        theta_v = np.where(bdy_halo_c == 1, exner, theta_v)
        exner = np.where(
            bdy_halo_c == 1, np.exp(rd_o_cvd * np.log(rd_o_p0ref * rho * exner)), exner
        )

        return dict(theta_v=theta_v, exner=exner)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rd_o_cvd = ta.wpfloat("10.0")
        rd_o_p0ref = ta.wpfloat("20.0")
        bdy_halo_c = data_alloc.random_mask(grid, dims.CellDim)
        exner = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=ta.wpfloat
        )
        rho = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=ta.wpfloat
        )
        theta_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=1, high=2, dtype=ta.wpfloat
        )

        return dict(
            bdy_halo_c=bdy_halo_c,
            rho=rho,
            theta_v=theta_v,
            exner=exner,
            rd_o_cvd=rd_o_cvd,
            rd_o_p0ref=rd_o_p0ref,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
