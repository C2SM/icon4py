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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import ice_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestIceToGraupel(stencil_tests.StencilTest):
    PROGRAM = ice_to_graupel
    OUTPUTS = ("aggregation",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        rho: np.ndarray,
        qr: np.ndarray,
        qg: np.ndarray,
        qi: np.ndarray,
        sticking_eff: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(aggregation=np.full(rho.shape, 7.1049436957697864e-19))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            rho=data_alloc.constant_field(1.04848, dims.CellDim, dims.KDim, dtype=wpfloat),
            qr=data_alloc.constant_field(6.00408e-13, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=data_alloc.constant_field(1.19022e-18, dims.CellDim, dims.KDim, dtype=wpfloat),
            qi=data_alloc.constant_field(1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            sticking_eff=data_alloc.constant_field(
                1.9584e-08, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            aggregation=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
