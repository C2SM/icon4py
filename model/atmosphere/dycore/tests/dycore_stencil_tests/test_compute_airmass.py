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

from icon4py.model.atmosphere.dycore.stencils.compute_airmass import compute_airmass
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils.data_allocation import random_field
from icon4py.model.testing.helpers import StencilTest


class TestComputeAirmass(StencilTest):
    PROGRAM = compute_airmass
    OUTPUTS = ("airmass_out",)

    @staticmethod
    def reference(
        grid, rho_in: np.array, ddqz_z_full_in: np.array, deepatmo_t1mc_in: np.array, **kwargs
    ) -> dict:
        airmass_out = rho_in * ddqz_z_full_in * deepatmo_t1mc_in
        return dict(airmass_out=airmass_out)

    @pytest.fixture
    def input_data(self, grid):
        rho_in = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        ddqz_z_full_in = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        deepatmo_t1mc_in = random_field(grid, dims.KDim, dtype=wpfloat)
        airmass_out = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        return dict(
            rho_in=rho_in,
            ddqz_z_full_in=ddqz_z_full_in,
            deepatmo_t1mc_in=deepatmo_t1mc_in,
            airmass_out=airmass_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
