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

from icon4py.model.atmosphere.dycore.stencils.compute_solver_coefficients_matrix import (
    compute_solver_coefficients_matrix,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def compute_solver_coefficients_matrix_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    current_exner: np.ndarray,
    current_rho: np.ndarray,
    current_theta_v: np.ndarray,
    inv_ddqz_z_full: np.ndarray,
    exner_w_implicit_weight_parameter: np.ndarray,
    theta_v_at_cells_on_half_levels: np.ndarray,
    rho_at_cells_on_half_levels: np.ndarray,
    dtime: float,
    rd: ta.wpfloat,
    cvd: ta.wpfloat,
) -> tuple[np.ndarray, np.ndarray]:
    tridiagonal_beta_coeff_at_cells_on_model_levels = dtime * rd * current_exner / (cvd * current_rho * current_theta_v) * inv_ddqz_z_full
    exner_w_implicit_weight_parameter = np.expand_dims(exner_w_implicit_weight_parameter, axis=-1)
    tridiagonal_alpha_coeff_at_cells_on_half_levels = exner_w_implicit_weight_parameter * theta_v_at_cells_on_half_levels * rho_at_cells_on_half_levels
    return (tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels)


class TestComputeSolverCoefficientsMatrix(StencilTest):
    PROGRAM = compute_solver_coefficients_matrix
    OUTPUTS = ("tridiagonal_beta_coeff_at_cells_on_model_levels", "tridiagonal_alpha_coeff_at_cells_on_half_levels")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        current_exner: np.ndarray,
        current_rho: np.ndarray,
        current_theta_v: np.ndarray,
        inv_ddqz_z_full: np.ndarray,
        exner_w_implicit_weight_parameter: np.ndarray,
        theta_v_at_cells_on_half_levels: np.ndarray,
        rho_at_cells_on_half_levels: np.ndarray,
        dtime: float,
        rd: ta.wpfloat,
        cvd: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        (tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels) = compute_solver_coefficients_matrix_numpy(
            connectivities,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            inv_ddqz_z_full=inv_ddqz_z_full,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
        )
        return dict(tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels, tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        current_exner = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        current_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        current_theta_v = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        inv_ddqz_z_full = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        exner_w_implicit_weight_parameter = data_alloc.random_field(grid, dims.CellDim, dtype=ta.wpfloat)
        theta_v_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        rho_at_cells_on_half_levels = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        tridiagonal_alpha_coeff_at_cells_on_half_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        tridiagonal_beta_coeff_at_cells_on_model_levels = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        dtime = ta.wpfloat("10.0")
        rd = ta.wpfloat("5.0")
        cvd = ta.wpfloat("3.0")

        return dict(
            tridiagonal_beta_coeff_at_cells_on_model_levels=tridiagonal_beta_coeff_at_cells_on_model_levels,
            current_exner=current_exner,
            current_rho=current_rho,
            current_theta_v=current_theta_v,
            inv_ddqz_z_full=inv_ddqz_z_full,
            tridiagonal_alpha_coeff_at_cells_on_half_levels=tridiagonal_alpha_coeff_at_cells_on_half_levels,
            exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            dtime=dtime,
            rd=rd,
            cvd=cvd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
