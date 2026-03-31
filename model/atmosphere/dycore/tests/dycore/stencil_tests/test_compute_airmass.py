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

from icon4py.model.atmosphere.dycore.stencils.compute_airmass import compute_airmass
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestComputeAirmass(StencilTest):
    PROGRAM = compute_airmass
    OUTPUTS = ("airmass_out",)

    @static_reference
    def reference(
        grid: base.Grid,
        rho_in: np.ndarray,
        ddqz_z_full_in: np.ndarray,
        deepatmo_t1mc_in: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        airmass_out = rho_in * ddqz_z_full_in * deepatmo_t1mc_in
        return dict(airmass_out=airmass_out)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_in = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        ddqz_z_full_in = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
        deepatmo_t1mc_in = self.data_alloc.random_field(dims.KDim, dtype=wpfloat)
        airmass_out = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=wpfloat)
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
