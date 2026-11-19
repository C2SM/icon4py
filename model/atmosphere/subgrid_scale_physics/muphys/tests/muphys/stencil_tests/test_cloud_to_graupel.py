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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_graupel
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestCloudToGraupel(stencil_tests.StencilTest):
    PROGRAM = cloud_to_graupel
    OUTPUTS = ("riming_graupel_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        t: np.ndarray,
        rho: np.ndarray,
        qc: np.ndarray,
        qg: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(riming_graupel_rate=np.full(t.shape, 2.7054723496793982e-10))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            t=data_alloc.constant_field(256.983, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(0.909677, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(8.60101e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            qg=data_alloc.constant_field(4.11575e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_graupel_rate=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
