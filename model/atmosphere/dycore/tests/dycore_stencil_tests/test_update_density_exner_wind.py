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

from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    update_density_exner_wind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.helpers import StencilTest


class TestUpdateDensityExnerWind(StencilTest):
    PROGRAM = update_density_exner_wind
    OUTPUTS = ("rho_new", "exner_new", "w_new")

    @staticmethod
    def reference(
        grid,
        rho_now: np.array,
        grf_tend_rho: np.array,
        theta_v_now: np.array,
        grf_tend_thv: np.array,
        w_now: np.array,
        grf_tend_w: np.array,
        dtime,
        **kwargs,
    ) -> tuple[np.array]:
        rho_new = rho_now + dtime * grf_tend_rho
        exner_new = theta_v_now + dtime * grf_tend_thv
        w_new = w_now + dtime * grf_tend_w
        return dict(rho_new=rho_new, exner_new=exner_new, w_new=w_new)

    @pytest.fixture
    def input_data(self, grid):
        rho_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        grf_tend_rho = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        theta_v_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        grf_tend_thv = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_now = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        grf_tend_w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        dtime = wpfloat("5.0")
        rho_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        exner_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        w_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)

        return dict(
            rho_now=rho_now,
            grf_tend_rho=grf_tend_rho,
            theta_v_now=theta_v_now,
            grf_tend_thv=grf_tend_thv,
            w_now=w_now,
            grf_tend_w=grf_tend_w,
            rho_new=rho_new,
            exner_new=exner_new,
            w_new=w_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
