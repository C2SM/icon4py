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

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import T_from_internal_energy
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing import stencil_tests


class TestTFromInternalEnergy(stencil_tests.StencilTest):
    PROGRAM = T_from_internal_energy
    OUTPUTS = ("temperature",)

    @stencil_tests.static_reference
    def reference(
        grid: base.Grid,
        *,
        u: np.ndarray,
        qv: np.ndarray,
        qliq: np.ndarray,
        qice: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(temperature=np.full(u.shape, 255.75599999999997))

    @stencil_tests.input_data_fixture
    def input_data(data_alloc: stencil_tests.DataAllocationWrapper):
        return dict(
            u=data_alloc.constant_field(38265357.270336017, dims.CellDim, dims.KDim, dtype=wpfloat),
            qv=data_alloc.constant_field(0.00122576, dims.CellDim, dims.KDim, dtype=wpfloat),
            qliq=data_alloc.constant_field(1.63837e-20, dims.CellDim, dims.KDim, dtype=wpfloat),
            qice=data_alloc.constant_field(1.09462e-08, dims.CellDim, dims.KDim, dtype=wpfloat),
            rho=data_alloc.constant_field(0.83444, dims.CellDim, dims.KDim, dtype=wpfloat),
            dz=data_alloc.constant_field(249.569, dims.CellDim, dims.KDim, dtype=wpfloat),
            temperature=data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
