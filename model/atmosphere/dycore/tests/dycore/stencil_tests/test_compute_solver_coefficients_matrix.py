# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Mapping
from typing import Any, cast

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.dycore.stencils.compute_solver_coefficients_matrix import (
    compute_solver_coefficients_matrix,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.testing.stencil_tests import StencilTest, input_data_fixture, static_reference


def compute_solver_coefficients_matrix_numpy(
    connectivities: Mapping[gtx.Dimension, np.ndarray],
    exner_nnow: np.ndarray,
    rho_nnow: np.ndarray,
    theta_v_nnow: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    vwind_impl_wgt: np.ndarray,
    theta_v_ic: np.ndarray,
    rho_ic: np.ndarray,
    dtime: float,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
) -> tuple[np.ndarray, np.ndarray]:
    z_beta = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full
    vwind_impl_wgt = np.expand_dims(vwind_impl_wgt, axis=-1)
    z_alpha = vwind_impl_wgt * theta_v_ic * rho_ic
    return (z_beta, z_alpha)


class TestComputeSolverCoefficientsMatrix(StencilTest):
    PROGRAM = compute_solver_coefficients_matrix
    OUTPUTS = ("z_beta", "z_alpha")

    @static_reference
    def reference(
        grid: base.Grid,
        exner_nnow: np.ndarray,
        rho_nnow: np.ndarray,
        theta_v_nnow: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        vwind_impl_wgt: np.ndarray,
        theta_v_ic: np.ndarray,
        rho_ic: np.ndarray,
        dtime: float,
        rd: ta.wpfloat,
        cvd: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        connectivities = cast(Mapping[gtx.Dimension, np.ndarray], grid.connectivities_asnumpy)
        (z_beta, z_alpha) = compute_solver_coefficients_matrix_numpy(
            connectivities,
            exner_nnow=exner_nnow,
            rho_nnow=rho_nnow,
            theta_v_nnow=theta_v_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
        )
        return dict(z_beta=z_beta, z_alpha=z_alpha)

    @input_data_fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        exner_nnow = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_nnow = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v_nnow = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        vwind_impl_wgt = self.data_alloc.random_field(dims.CellDim, dtype=ta.wpfloat)
        theta_v_ic = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_ic = self.data_alloc.random_field(dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        z_alpha = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_beta = self.data_alloc.zero_field(dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        dtime = ta.wpfloat("10.0")
        rd = ta.wpfloat("5.0")
        cvd = ta.wpfloat("3.0")

        return dict(
            z_beta=z_beta,
            exner_nnow=exner_nnow,
            rho_nnow=rho_nnow,
            theta_v_nnow=theta_v_nnow,
            inv_ddqz_z_full=inv_ddqz_z_full,
            z_alpha=z_alpha,
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            rho_ic=rho_ic,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
