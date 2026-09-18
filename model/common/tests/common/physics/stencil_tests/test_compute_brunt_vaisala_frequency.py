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

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.physics.compute_brunt_vaisala_frequency import (
    _compute_brunt_vaisala_frequency,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import reference_funcs, stencil_tests


class TestComputeBruntVaisalaFrequency(stencil_tests.StencilTest):
    PROGRAM = _compute_brunt_vaisala_frequency
    OUTPUTS = ("out",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        theta_v: np.ndarray,
        wgtfac_c: np.ndarray,
        inv_ddqz_z_half: np.ndarray,
        grav: float,
        **kwargs,
    ) -> dict:
        bruvais = reference_funcs.compute_brunt_vaisala_frequency_numpy(
            theta_v, wgtfac_c, inv_ddqz_z_half, grav=grav
        )
        return dict(out=bruvais)

    @stencil_tests.input_data_fixture
    def input_data(
        data_alloc: stencil_tests.DataAllocationWrapper, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType]:
        theta_v = data_alloc.random_field(
            dims.CellDim, dims.KDim, low=270.0, high=350.0, dtype=wpfloat
        )
        wgtfac_c = data_alloc.random_field(
            dims.CellDim, dims.KHalfDim, low=0.0, high=1.0, dtype=wpfloat
        )
        inv_ddqz_z_half = data_alloc.random_field(
            dims.CellDim,
            dims.KHalfDim,
            low=1.0e-3,
            high=1.0e-1,
            dtype=wpfloat,
        )
        return dict(
            theta_v=theta_v,
            wgtfac_c=wgtfac_c,
            inv_ddqz_z_half=inv_ddqz_z_half,
            grav=constants.GRAV,
            # Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based half levels)
            domain={
                dims.CellDim: (0, gtx.int32(grid.num_cells)),
                dims.KHalfDim: (1, gtx.int32(grid.num_levels)),
            },
            out=data_alloc.zero_field(dims.CellDim, dims.KHalfDim, dtype=wpfloat),
        )
