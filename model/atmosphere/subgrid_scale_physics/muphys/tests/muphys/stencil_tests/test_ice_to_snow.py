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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestIceToSnow(stencil_tests.StencilTest):
    PROGRAM = ice_to_snow
    OUTPUTS = ("conversion_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        qi: np.ndarray,
        ns: np.ndarray,
        lam: np.ndarray,
        sticking_eff: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(conversion_rate=np.full(qi.shape, 3.3262745200740486e-11))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            qi=data_alloc.constant_field(6.43223e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=data_alloc.constant_field(1.93157e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=data_alloc.constant_field(10576.8, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff=data_alloc.constant_field(
                0.511825, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            conversion_rate=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
