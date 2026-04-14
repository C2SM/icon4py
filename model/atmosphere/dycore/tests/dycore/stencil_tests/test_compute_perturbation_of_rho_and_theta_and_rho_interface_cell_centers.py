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

from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers import (
    compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy(
    wgtfac_c: np.ndarray,
    rho: np.ndarray,
    reference_rho_at_cells_on_model_levels: np.ndarray,
    theta_v: np.ndarray,
    reference_theta_at_cells_on_model_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_offset_1 = np.roll(rho, shift=1, axis=1)
    rho_at_cells_on_half_levels = wgtfac_c * rho + (1.0 - wgtfac_c) * rho_offset_1
    rho_at_cells_on_half_levels[:, 0] = 0
    z_rth_pr_1 = rho - reference_rho_at_cells_on_model_levels
    z_rth_pr_1[:, 0] = 0
    perturbed_theta_v_at_cells_on_model_levels_2 = theta_v - reference_theta_at_cells_on_model_levels
    perturbed_theta_v_at_cells_on_model_levels_2[:, 0] = 0
    return rho_at_cells_on_half_levels, z_rth_pr_1, perturbed_theta_v_at_cells_on_model_levels_2


class TestComputePerturbationOfRhoAndThetaAndRhoInterfaceCellCenters(StencilTest):
    PROGRAM = compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers
    OUTPUTS = ("rho_at_cells_on_half_levels", "z_rth_pr_1", "perturbed_theta_v_at_cells_on_model_levels_2")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        wgtfac_c: np.ndarray,
        rho: np.ndarray,
        reference_rho_at_cells_on_model_levels: np.ndarray,
        theta_v: np.ndarray,
        reference_theta_at_cells_on_model_levels: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (
            rho_at_cells_on_half_levels,
            z_rth_pr_1,
            perturbed_theta_v_at_cells_on_model_levels_2,
        ) = compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers_numpy(
            wgtfac_c=wgtfac_c,
            rho=rho,
            reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
            theta_v=theta_v,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        )
        return dict(rho_at_cells_on_half_levels=rho_at_cells_on_half_levels, z_rth_pr_1=z_rth_pr_1, perturbed_theta_v_at_cells_on_model_levels_2=perturbed_theta_v_at_cells_on_model_levels_2)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        wgtfac_c = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        rho = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        reference_rho_at_cells_on_model_levels = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        theta_v = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        reference_theta_at_cells_on_model_levels = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        rho_at_cells_on_half_levels = zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_rth_pr_1 = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        perturbed_theta_v_at_cells_on_model_levels_2 = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            wgtfac_c=wgtfac_c,
            rho=rho,
            reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
            theta_v=theta_v,
            reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
            rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
            z_rth_pr_1=z_rth_pr_1,
            perturbed_theta_v_at_cells_on_model_levels_2=perturbed_theta_v_at_cells_on_model_levels_2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
