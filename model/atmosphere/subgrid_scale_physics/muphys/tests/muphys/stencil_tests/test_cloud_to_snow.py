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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.transitions import cloud_to_snow
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestCloudToSnow(stencil_tests.StencilTest):
    PROGRAM = cloud_to_snow
    OUTPUTS = ("riming_snow_rate",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        t: np.ndarray,
        qc: np.ndarray,
        qs: np.ndarray,
        ns: np.ndarray,
        lam: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(riming_snow_rate=np.full(t.shape, 9.5431874564438999e-10))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            t=data_alloc.constant_field(256.571, dims.CellDim, dims.KDim, dtype=wpfloat),
            qc=data_alloc.constant_field(3.31476e-05, dims.CellDim, dims.KDim, dtype=wpfloat),
            qs=data_alloc.constant_field(7.47365e-06, dims.CellDim, dims.KDim, dtype=wpfloat),
            ns=data_alloc.constant_field(3.37707e07, dims.CellDim, dims.KDim, dtype=wpfloat),
            lam=data_alloc.constant_field(8989.78, dims.CellDim, dims.KDim, dtype=wpfloat),
            riming_snow_rate=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
