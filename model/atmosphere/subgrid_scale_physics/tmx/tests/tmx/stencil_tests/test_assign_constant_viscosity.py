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

from icon4py.model.atmosphere.subgrid_scale_physics.tmx.stencils.assign_constant_viscosity import (
    assign_constant_viscosity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def assign_constant_viscosity_numpy(
    rho_ic: np.ndarray, km_const: float, rturb_prandtl: float
) -> tuple[np.ndarray, np.ndarray]:
    nlev = rho_ic.shape[1] - 1
    km_ic = np.zeros_like(rho_ic)
    # interior half levels, Fortran jk = 2..nlev (1-based) -> k = 1..nlev-1 (0-based)
    km_ic[:, 1:nlev] = rho_ic[:, 1:nlev] * km_const
    # boundary rows are copies of the adjacent interior rows
    # (Fortran 1-based: k = 1 <- k = 2, k = nlevp1 <- k = nlev)
    km_ic[:, 0] = km_ic[:, 1]
    km_ic[:, nlev] = km_ic[:, nlev - 1]
    kh_ic = km_ic * rturb_prandtl
    return km_ic, kh_ic


class TestAssignConstantViscosity(StencilTest):
    PROGRAM = assign_constant_viscosity
    OUTPUTS = ("km_ic", "kh_ic")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_ic: np.ndarray,
        km_const: float,
        rturb_prandtl: float,
        **kwargs,
    ) -> dict:
        km_ic, kh_ic = assign_constant_viscosity_numpy(rho_ic, km_const, rturb_prandtl)
        return dict(km_ic=km_ic, kh_ic=kh_ic)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_ic = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, low=0.5, high=1.4, dtype=wpfloat, extend={dims.KDim: 1}
        )
        km_ic = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )
        kh_ic = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=wpfloat, extend={dims.KDim: 1}
        )

        return dict(
            rho_ic=rho_ic,
            km_ic=km_ic,
            kh_ic=kh_ic,
            km_const=wpfloat(0.05),
            rturb_prandtl=wpfloat(2.0),
            nlev=gtx.int32(grid.num_levels),
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels + 1),
        )
