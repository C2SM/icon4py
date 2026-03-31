# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.thermo import internal_energy
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


class TestInternalEnergy(StencilTest):
    PROGRAM = internal_energy
    OUTPUTS = ("energy",)

    @static_reference
    def reference(
        grid: base.Grid,
        t: np.ndarray,
        qv: np.ndarray,
        qliq: np.ndarray,
        qice: np.ndarray,
        rho: np.ndarray,
        dz: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(energy=np.full(t.shape, 38265357.270336017))

    @input_data_fixture
    def input_data(self, grid: base.Grid):
        return dict(
            t=self.data_alloc.constant_field(255.756, dims.CellDim, dims.KDim, dtype=wpfloat),
            qv=self.data_alloc.constant_field(0.00122576, dims.CellDim, dims.KDim, dtype=wpfloat),
            qliq=self.data_alloc.constant_field(
                1.63837e-20, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            qice=self.data_alloc.constant_field(
                1.09462e-08, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            rho=self.data_alloc.constant_field(0.83444, dims.CellDim, dims.KDim, dtype=wpfloat),
            dz=self.data_alloc.constant_field(249.569, dims.CellDim, dims.KDim, dtype=wpfloat),
            energy=self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=wpfloat),
        )
